from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np

from wc_forecast.data import Match
from wc_forecast.features import WORLD_CUP_TOURNAMENTS
from wc_forecast.models import ForecastModel

GROUP_SIZE = 4
GROUP_MATCHES_PER_GROUP = 6  # round-robin between 4 teams
GROUP_NAMES = "ABCDEFGHIJKLMNOPQRSTUVWX"


@dataclass
class TournamentSpec:
    groups: Dict[str, List[str]]
    hosts: Set[str]

    @property
    def teams(self) -> List[str]:
        return [team for group in self.groups.values() for team in group]


def world_cup_matches(matches: List[Match], year: int) -> List[Match]:
    """Return the finals matches of the World Cup held in `year`, in date order."""
    selected = [
        match
        for match in matches
        if match.date.year == year and match.tournament.strip().lower() in WORLD_CUP_TOURNAMENTS
    ]
    return sorted(selected, key=lambda match: match.date)


def infer_spec(wc_matches: List[Match]) -> TournamentSpec:
    """Reconstruct the group draw and hosts from a tournament's played matches.

    The group stage is the first chronological block of matches; teams that met
    in it form connected components of exactly GROUP_SIZE teams.
    """
    teams = sorted({team for match in wc_matches for team in (match.home_team, match.away_team)})
    if not teams or len(teams) % GROUP_SIZE:
        raise ValueError(f"Expected groups of {GROUP_SIZE} teams, found {len(teams)} teams.")
    group_count = len(teams) // GROUP_SIZE
    group_stage = wc_matches[: group_count * GROUP_MATCHES_PER_GROUP]

    parent = {team: team for team in teams}

    def find(team: str) -> str:
        while parent[team] != team:
            parent[team] = parent[parent[team]]
            team = parent[team]
        return team

    for match in group_stage:
        parent[find(match.home_team)] = find(match.away_team)

    components: Dict[str, List[str]] = {}
    for team in teams:
        components.setdefault(find(team), []).append(team)
    if len(components) != group_count or any(len(members) != GROUP_SIZE for members in components.values()):
        raise ValueError(
            "Could not reconstruct groups: the group stage does not split into "
            f"{group_count} groups of {GROUP_SIZE}. This tournament format is unsupported."
        )

    # Name groups A, B, ... in the order they first appear in the schedule.
    ordered_roots: List[str] = []
    for match in group_stage:
        root = find(match.home_team)
        if root not in ordered_roots:
            ordered_roots.append(root)
    groups = {GROUP_NAMES[idx]: sorted(components[root]) for idx, root in enumerate(ordered_roots)}

    hosts = {match.home_team for match in wc_matches if not match.neutral}
    return TournamentSpec(groups=groups, hosts=hosts)


def knockout_bracket_size(group_count: int) -> int:
    """Direct qualifiers (top two per group) rounded up to a power of two,
    with the gap filled by best third-placed teams — matches FIFA formats."""
    direct = 2 * group_count
    size = 1
    while size < direct:
        size *= 2
    return size


def matchup_probabilities(model: ForecastModel, spec: TournamentSpec) -> np.ndarray:
    """Probability tensor P[i, j] = [p(team i wins), p(draw), p(team j wins)].

    A host playing a non-host gets home advantage; all other matchups are neutral.
    """
    teams = spec.teams
    count = len(teams)
    pairs = [(i, j) for i in range(count) for j in range(i + 1, count)]
    matchups = []
    swapped = []
    for i, j in pairs:
        i_host = teams[i] in spec.hosts
        j_host = teams[j] in spec.hosts
        if j_host and not i_host:
            matchups.append((teams[j], teams[i], False, "FIFA World Cup"))
            swapped.append(True)
        else:
            matchups.append((teams[i], teams[j], not (i_host and not j_host), "FIFA World Cup"))
            swapped.append(False)

    raw = model.predict_proba_batch(matchups)  # columns [away_win, draw, home_win]
    probs = np.zeros((count, count, 3))
    for (i, j), row, was_swapped in zip(pairs, raw, swapped):
        if was_swapped:  # team j was the home side, so p(i wins) is the away column
            probs[i, j] = [row[0], row[1], row[2]]
            probs[j, i] = [row[2], row[1], row[0]]
        else:
            probs[i, j] = [row[2], row[1], row[0]]
            probs[j, i] = [row[0], row[1], row[2]]
    return probs


def simulate_tournament(
    model: ForecastModel,
    spec: TournamentSpec,
    runs: int = 5000,
    seed: int = 7,
) -> List[dict]:
    """Monte Carlo the whole tournament; returns per-team advancement probabilities.

    Approximations: knockout pairings re-seed each round by group-stage record
    (winners first, then runners-up, then thirds), and a drawn knockout match is
    decided by a 50/50 shootout.
    """
    if runs <= 0:
        raise ValueError("runs must be a positive integer")

    rng = np.random.default_rng(seed)
    teams = spec.teams
    index = {team: position for position, team in enumerate(teams)}
    probs = matchup_probabilities(model, spec)
    elo = np.array([model.team_states[team].elo if team in model.team_states else 1500.0 for team in teams])

    group_indices = [[index[team] for team in members] for members in spec.groups.values()]
    bracket_size = knockout_bracket_size(len(group_indices))
    third_slots = bracket_size - 2 * len(group_indices)

    champion = np.zeros(len(teams))
    finalist = np.zeros(len(teams))
    semifinalist = np.zeros(len(teams))

    for _ in range(runs):
        winners: List[tuple] = []
        runners: List[tuple] = []
        thirds: List[tuple] = []

        for members in group_indices:
            points = {team: 0 for team in members}
            for a_pos in range(GROUP_SIZE):
                for b_pos in range(a_pos + 1, GROUP_SIZE):
                    a, b = members[a_pos], members[b_pos]
                    p_a, p_draw, _ = probs[a, b]
                    draw = rng.random()
                    if draw < p_a:
                        points[a] += 3
                    elif draw < p_a + p_draw:
                        points[a] += 1
                        points[b] += 1
                    else:
                        points[b] += 3
            # Rank on points with Elo as the tiebreaker (a stand-in for goal difference).
            table = sorted(members, key=lambda team: (points[team], elo[team]), reverse=True)
            winners.append((points[table[0]], elo[table[0]], table[0]))
            runners.append((points[table[1]], elo[table[1]], table[1]))
            thirds.append((points[table[2]], elo[table[2]], table[2]))

        qualifiers = [entry[2] for entry in sorted(winners, reverse=True)]
        qualifiers += [entry[2] for entry in sorted(runners, reverse=True)]
        qualifiers += [entry[2] for entry in sorted(thirds, reverse=True)[:third_slots]]

        stage = qualifiers
        while len(stage) > 1:
            if len(stage) == 4:
                semifinalist[stage] += 1
            if len(stage) == 2:
                finalist[stage] += 1
            survivors = []
            for position in range(len(stage) // 2):
                a, b = stage[position], stage[len(stage) - 1 - position]
                p_a_wins = probs[a, b, 0] + 0.5 * probs[a, b, 1]
                survivors.append(a if rng.random() < p_a_wins else b)
            stage = survivors
        champion[stage[0]] += 1

    rows = [
        {
            "team": team,
            "champion": champion[position] / runs,
            "finalist": finalist[position] / runs,
            "semifinalist": semifinalist[position] / runs,
        }
        for position, team in enumerate(teams)
    ]
    return sorted(rows, key=lambda row: (row["champion"], row["finalist"], row["semifinalist"]), reverse=True)
