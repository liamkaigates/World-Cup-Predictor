const state = {
  teams: [],
  summary: null,
};

const labels = {
  away_win: "Away win",
  draw: "Draw",
  home_win: "Home win",
};

function percent(value) {
  return `${Math.round(value * 100)}%`;
}

async function getJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Request failed: ${path}`);
  }
  return response.json();
}

function setStatus(text) {
  document.getElementById("status").textContent = text;
}

function fillTeamSelect(select, includeAll = false) {
  select.innerHTML = "";
  if (includeAll) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "All teams";
    select.appendChild(option);
  }
  state.teams.forEach((team) => {
    const option = document.createElement("option");
    option.value = team;
    option.textContent = team;
    select.appendChild(option);
  });
}

function renderSummary(summary) {
  document.getElementById("matchCount").textContent = summary.match_count;
  document.getElementById("teamCount").textContent = summary.team_count;
  document.getElementById("yearRange").textContent = `${summary.first_year}-${summary.last_year}`;
  document.getElementById("goalsPerMatch").textContent = summary.goals_per_match.toFixed(2);
  renderBars(document.getElementById("outcomeBars"), summary.outcomes, summary.match_count);
  renderAttackChart(summary.top_attack);
}

function renderBars(container, values, total = 1) {
  container.innerHTML = "";
  Object.entries(values).forEach(([key, raw]) => {
    const value = total === 1 ? raw : raw / total;
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <span>${labels[key] || key}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.max(value * 100, 2)}%"></div></div>
      <strong>${percent(value)}</strong>
    `;
    container.appendChild(row);
  });
}

function renderAttackChart(rows) {
  const container = document.getElementById("attackChart");
  container.innerHTML = "";
  const max = Math.max(...rows.map((row) => row.goals_per_match), 1);
  rows.forEach((row) => {
    const item = document.createElement("div");
    item.className = "rank-row";
    item.innerHTML = `
      <span>${row.team}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${(row.goals_per_match / max) * 100}%"></div></div>
      <strong>${row.goals_per_match.toFixed(2)}</strong>
    `;
    container.appendChild(item);
  });
}

async function predict() {
  const home = document.getElementById("homeTeam").value;
  const away = document.getElementById("awayTeam").value;
  const neutral = document.getElementById("neutral").checked;
  if (home === away) {
    setStatus("Choose two teams");
    return;
  }
  const result = await getJson(`/api/predict?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}&neutral=${neutral}`);
  document.getElementById("predictionLabel").textContent = labels[result.prediction] || result.prediction;
  renderBars(document.getElementById("probabilityBars"), result.probabilities);
  setStatus("Prediction ready");
}

async function loadMatches() {
  const team = document.getElementById("historyTeam").value;
  const payload = await getJson(`/api/matches?team=${encodeURIComponent(team)}&limit=60`);
  const tbody = document.getElementById("matchesTable");
  tbody.innerHTML = "";
  payload.matches.forEach((match) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${match.date}</td>
      <td>${match.home_team} vs ${match.away_team}</td>
      <td>${match.score}</td>
      <td>${match.neutral ? "Neutral" : "Host"}</td>
    `;
    tbody.appendChild(row);
  });
}

function bindEvents() {
  document.getElementById("predict").addEventListener("click", predict);
  document.getElementById("historyTeam").addEventListener("change", loadMatches);
  document.getElementById("swapTeams").addEventListener("click", () => {
    const home = document.getElementById("homeTeam");
    const away = document.getElementById("awayTeam");
    const oldHome = home.value;
    home.value = away.value;
    away.value = oldHome;
    predict();
  });
}

async function init() {
  try {
    const [teams, summary] = await Promise.all([getJson("/api/teams"), getJson("/api/summary")]);
    state.teams = teams.teams;
    state.summary = summary;

    fillTeamSelect(document.getElementById("homeTeam"));
    fillTeamSelect(document.getElementById("awayTeam"));
    fillTeamSelect(document.getElementById("historyTeam"), true);

    const argentina = state.teams.indexOf("Argentina");
    const france = state.teams.indexOf("France");
    document.getElementById("homeTeam").selectedIndex = argentina >= 0 ? argentina : 0;
    document.getElementById("awayTeam").selectedIndex = france >= 0 ? france : Math.min(1, state.teams.length - 1);

    renderSummary(summary);
    bindEvents();
    await Promise.all([predict(), loadMatches()]);
  } catch (error) {
    console.error(error);
    setStatus("Dashboard error");
  }
}

init();
