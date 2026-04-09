# -*- coding: utf-8 -*-
import math
from io import BytesIO
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="3자 경선 시뮬레이터", layout="wide")

CANDIDATES = ["김", "조", "이"]


def normalize_poll_scores(poll_scores: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(v, 0.0) for v in poll_scores.values())
    if total <= 0:
        return poll_scores.copy()
    return {k: (max(v, 0.0) / total) * 100 for k, v in poll_scores.items()}


def calculate_actual_voters(total_members: int, loss_rate_pct: float, turnout_rate_pct: float) -> float:
    valid_members = total_members * (1 - loss_rate_pct / 100)
    actual_voters = valid_members * (turnout_rate_pct / 100)
    return actual_voters


def convert_party_input_to_votes(
    input_mode: str,
    actual_voters: float,
    vote_inputs: Dict[str, int],
    rate_inputs: Dict[str, float],
    normalize_party_rate: bool,
) -> Dict[str, int]:
    if input_mode == "표(숫자)":
        return {name: int(vote_inputs[name]) for name in CANDIDATES}

    raw_rates = {name: max(0.0, rate_inputs[name]) for name in CANDIDATES}
    total_rate = sum(raw_rates.values())

    if total_rate <= 0:
        raise ValueError("책임당원 득표율 입력 합계가 0입니다.")

    if normalize_party_rate:
        normalized_rates = {name: (raw_rates[name] / total_rate) * 100 for name in CANDIDATES}
    else:
        normalized_rates = raw_rates.copy()

    votes_float = {name: actual_voters * (normalized_rates[name] / 100) for name in CANDIDATES}
    votes_int = {name: int(math.floor(v)) for name, v in votes_float.items()}

    allocated = sum(votes_int.values())
    remainder = int(round(actual_voters)) - allocated

    if remainder > 0:
        fractions = sorted(
            [(name, votes_float[name] - votes_int[name]) for name in CANDIDATES],
            key=lambda x: x[1],
            reverse=True,
        )
        for i in range(remainder):
            votes_int[fractions[i % len(fractions)][0]] += 1

    return votes_int


def calculate_simulation(
    total_members: int,
    loss_rate_pct: float,
    turnout_rate_pct: float,
    poll_weight_pct: float,
    party_weight_pct: float,
    poll_scores: Dict[str, float],
    party_votes: Dict[str, int],
    normalize_poll: bool,
) -> Dict[str, object]:
    if abs((poll_weight_pct + party_weight_pct) - 100) > 1e-9:
        raise ValueError("여론 반영 비율과 책임당원 반영 비율의 합은 100이어야 합니다.")

    actual_voters = calculate_actual_voters(total_members, loss_rate_pct, turnout_rate_pct)
    if actual_voters <= 0:
        raise ValueError("실제 투표수가 0 이하입니다. 설정값을 확인하세요.")

    if normalize_poll:
        poll_scores = normalize_poll_scores(poll_scores)

    total_party_votes = sum(party_votes.values())
    rounded_actual_voters = int(round(actual_voters))
    remaining_votes = rounded_actual_voters - total_party_votes

    if total_party_votes > rounded_actual_voters:
        raise ValueError(
            f"책임당원 확보표 합계({total_party_votes:,.0f})가 실제 투표수({rounded_actual_voters:,.0f})보다 큽니다."
        )

    party_rates = {
        name: (party_votes[name] / actual_voters) * 100 for name in CANDIDATES
    }

    final_scores = {
        name: poll_scores[name] * (poll_weight_pct / 100)
        + party_rates[name] * (party_weight_pct / 100)
        for name in CANDIDATES
    }

    result_df = pd.DataFrame(
        {
            "후보": CANDIDATES,
            "일반여론(%)": [poll_scores[name] for name in CANDIDATES],
            "책임당원 확보표": [party_votes[name] for name in CANDIDATES],
            "책임당원 득표율(%)": [party_rates[name] for name in CANDIDATES],
            "최종점수": [final_scores[name] for name in CANDIDATES],
        }
    ).sort_values("최종점수", ascending=False, ignore_index=True)

    winner = result_df.loc[0, "후보"]

    return {
        "actual_voters": actual_voters,
        "valid_members": total_members * (1 - loss_rate_pct / 100),
        "total_party_votes": total_party_votes,
        "remaining_votes": remaining_votes,
        "poll_scores": poll_scores,
        "party_rates": party_rates,
        "final_scores": final_scores,
        "winner": winner,
        "result_df": result_df,
    }


def calculate_needed_votes_to_beat(
    target_name: str,
    opponent_name: str,
    actual_voters: float,
    target_poll: float,
    opponent_poll: float,
    opponent_votes: int,
    poll_weight_pct: float,
    party_weight_pct: float,
) -> int:
    if party_weight_pct <= 0:
        return math.inf

    opponent_party_rate = (opponent_votes / actual_voters) * 100
    opponent_final = (
        opponent_poll * (poll_weight_pct / 100)
        + opponent_party_rate * (party_weight_pct / 100)
    )

    needed_party_rate = (opponent_final - target_poll * (poll_weight_pct / 100)) / (party_weight_pct / 100)
    needed_votes = math.floor((needed_party_rate / 100) * actual_voters) + 1

    return max(0, needed_votes)


def build_scenario_table(
    total_members: int,
    loss_rate_pct: float,
    turnout_rate_pct: float,
    poll_weight_pct: float,
    party_weight_pct: float,
    poll_scores: Dict[str, float],
    jo_start: int,
    jo_end: int,
    jo_step: int,
    kim_share_of_remaining_pct: float,
) -> pd.DataFrame:
    actual_voters = calculate_actual_voters(total_members, loss_rate_pct, turnout_rate_pct)
    rounded_actual_voters = int(round(actual_voters))
    rows: List[Dict[str, object]] = []

    if jo_step <= 0:
        raise ValueError("조 표 범위 간격은 1 이상이어야 합니다.")

    current = jo_start
    while current >= jo_end:
        jo_votes = current
        remaining = max(0, rounded_actual_voters - jo_votes)
        kim_votes = int(round(remaining * (kim_share_of_remaining_pct / 100)))
        lee_votes = remaining - kim_votes

        party_votes = {"김": kim_votes, "조": jo_votes, "이": lee_votes}

        sim = calculate_simulation(
            total_members=total_members,
            loss_rate_pct=loss_rate_pct,
            turnout_rate_pct=turnout_rate_pct,
            poll_weight_pct=poll_weight_pct,
            party_weight_pct=party_weight_pct,
            poll_scores=poll_scores,
            party_votes=party_votes,
            normalize_poll=False,
        )

        rows.append(
            {
                "조 확보표": jo_votes,
                "김 확보표": kim_votes,
                "이 확보표": lee_votes,
                "실제 투표수": rounded_actual_voters,
                "조 책당율(%)": sim["party_rates"]["조"],
                "김 책당율(%)": sim["party_rates"]["김"],
                "이 책당율(%)": sim["party_rates"]["이"],
                "조 최종점수": sim["final_scores"]["조"],
                "김 최종점수": sim["final_scores"]["김"],
                "이 최종점수": sim["final_scores"]["이"],
                "승자": sim["winner"],
            }
        )

        current -= jo_step

    return pd.DataFrame(rows)


def make_bar_chart(df: pd.DataFrame, y_col: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["후보"], df[y_col])
    ax.set_title(title)
    ax.set_ylabel(y_col)
    ax.set_xlabel("후보")
    ax.grid(axis="y", alpha=0.25)
    return fig


def make_comparison_chart(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(df))
    width = 0.25

    ax.bar([i - width for i in x], df["일반여론(%)"], width=width, label="일반여론")
    ax.bar(x, df["책임당원 득표율(%)"], width=width, label="책당득표율")
    ax.bar([i + width for i in x], df["최종점수"], width=width, label="최종점수")

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["후보"])
    ax.set_ylabel("점수 / 비율")
    ax.set_title("후보별 비교")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    return fig


def dataframe_to_excel_bytes(main_df: pd.DataFrame, scenario_df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        main_df.to_excel(writer, index=False, sheet_name="현재시뮬레이션")
        scenario_df.to_excel(writer, index=False, sheet_name="조전략시나리오")
    return output.getvalue()


st.title("국민의힘 3자 경선 시뮬레이터")
st.caption("여론조사 점수와 책임당원표를 합산해 최종 승자를 계산합니다.")

with st.sidebar:
    st.header("기본 설정")
    total_members = st.number_input("전체 책임당원 수", min_value=1, value=5500, step=100)
    loss_rate_pct = st.number_input("전체 로스율(%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
    turnout_rate_pct = st.number_input("유효 당원 내 투표율(%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
    poll_weight_pct = st.number_input("여론 반영 비율(%)", min_value=0.0, max_value=100.0, value=50.0, step=5.0)
    party_weight_pct = st.number_input("책임당원 반영 비율(%)", min_value=0.0, max_value=100.0, value=50.0, step=5.0)
    normalize_poll = st.checkbox("3자 여론 합계를 100으로 재정규화", value=False)

st.subheader("1. 일반여론 입력")
col1, col2, col3 = st.columns(3)
with col1:
    poll_kim = st.number_input("김 여론(%)", min_value=0.0, value=40.0, step=0.1)
with col2:
    poll_jo = st.number_input("조 여론(%)", min_value=0.0, value=28.0, step=0.1)
with col3:
    poll_lee = st.number_input("이 여론(%)", min_value=0.0, value=24.0, step=0.1)

st.subheader("2. 책임당원 입력 방식")
input_mode = st.radio("입력 방식 선택", ["표(숫자)", "득표율(%)"], horizontal=True)

actual_voters_preview = calculate_actual_voters(total_members, loss_rate_pct, turnout_rate_pct)
st.info(f"현재 설정 기준 실제 책임당원 투표수: {int(round(actual_voters_preview)):,.0f}표")

vote_inputs = {"김": 0, "조": 0, "이": 0}
rate_inputs = {"김": 0.0, "조": 0.0, "이": 0.0}
normalize_party_rate = False

if input_mode == "표(숫자)":
    c1, c2, c3 = st.columns(3)
    with c1:
        vote_inputs["김"] = st.number_input("김 확보표", min_value=0, value=700, step=10)
    with c2:
        vote_inputs["조"] = st.number_input("조 확보표", min_value=0, value=2300, step=10)
    with c3:
        vote_inputs["이"] = st.number_input("이 확보표", min_value=0, value=520, step=10)
else:
    normalize_party_rate = st.checkbox("책임당원 득표율 합계를 100으로 자동 보정", value=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        rate_inputs["김"] = st.number_input("김 득표율(%)", min_value=0.0, value=20.0, step=0.1)
    with c2:
        rate_inputs["조"] = st.number_input("조 득표율(%)", min_value=0.0, value=65.0, step=0.1)
    with c3:
        rate_inputs["이"] = st.number_input("이 득표율(%)", min_value=0.0, value=15.0, step=0.1)

    total_rate_input = sum(rate_inputs.values())
    st.write(f"입력된 책임당원 득표율 합계: {total_rate_input:.1f}%")

st.subheader("3. 조 전략 시나리오 설정")
c4, c5, c6, c7 = st.columns(4)
with c4:
    jo_start = st.number_input("조 시작표", min_value=0, value=2500, step=100)
with c5:
    jo_end = st.number_input("조 끝표", min_value=0, value=1800, step=100)
with c6:
    jo_step = st.number_input("간격", min_value=1, value=100, step=10)
with c7:
    kim_share_of_remaining_pct = st.number_input("남은 표 중 김 비중(%)", min_value=0.0, max_value=100.0, value=60.0, step=5.0)

poll_scores = {"김": poll_kim, "조": poll_jo, "이": poll_lee}

try:
    party_votes = convert_party_input_to_votes(
        input_mode=input_mode,
        actual_voters=actual_voters_preview,
        vote_inputs=vote_inputs,
        rate_inputs=rate_inputs,
        normalize_party_rate=normalize_party_rate,
    )

    simulation = calculate_simulation(
        total_members=total_members,
        loss_rate_pct=loss_rate_pct,
        turnout_rate_pct=turnout_rate_pct,
        poll_weight_pct=poll_weight_pct,
        party_weight_pct=party_weight_pct,
        poll_scores=poll_scores,
        party_votes=party_votes,
        normalize_poll=normalize_poll,
    )

    actual_voters = simulation["actual_voters"]
    result_df = simulation["result_df"]

    a, b, c, d = st.columns(4)
    a.metric("로스 반영 후 유효 당원", f"{simulation['valid_members']:,.0f}명")
    b.metric("실제 책임당원 투표수", f"{actual_voters:,.0f}표")
    c.metric("입력된 확보표 합계", f"{simulation['total_party_votes']:,.0f}표")
    d.metric("남은 미배분 표", f"{simulation['remaining_votes']:,.0f}표")

    st.success(f"예상 승자: {simulation['winner']}")

    if input_mode == "득표율(%)":
        st.subheader("득표율 입력 → 실제 표 환산 결과")
        converted_df = pd.DataFrame(
            {
                "후보": CANDIDATES,
                "환산 확보표": [party_votes[name] for name in CANDIDATES],
                "환산 득표율(%)": [simulation["party_rates"][name] for name in CANDIDATES],
            }
        )
        converted_df["환산 득표율(%)"] = converted_df["환산 득표율(%)"].map(lambda x: round(float(x), 2))
        st.dataframe(converted_df, use_container_width=True)

    st.subheader("현재 시뮬레이션 결과")
    display_df = result_df.copy()
    for col in ["일반여론(%)", "책임당원 득표율(%)", "최종점수"]:
        display_df[col] = display_df[col].map(lambda x: round(float(x), 2))
    st.dataframe(display_df, use_container_width=True)

    st.subheader("승리 역산")
    jo_needed_to_beat_kim = calculate_needed_votes_to_beat(
        target_name="조",
        opponent_name="김",
        actual_voters=actual_voters,
        target_poll=simulation["poll_scores"]["조"],
        opponent_poll=simulation["poll_scores"]["김"],
        opponent_votes=party_votes["김"],
        poll_weight_pct=poll_weight_pct,
        party_weight_pct=party_weight_pct,
    )
    kim_needed_to_beat_jo = calculate_needed_votes_to_beat(
        target_name="김",
        opponent_name="조",
        actual_voters=actual_voters,
        target_poll=simulation["poll_scores"]["김"],
        opponent_poll=simulation["poll_scores"]["조"],
        opponent_votes=party_votes["조"],
        poll_weight_pct=poll_weight_pct,
        party_weight_pct=party_weight_pct,
    )
    lee_needed_to_beat_jo = calculate_needed_votes_to_beat(
        target_name="이",
        opponent_name="조",
        actual_voters=actual_voters,
        target_poll=simulation["poll_scores"]["이"],
        opponent_poll=simulation["poll_scores"]["조"],
        opponent_votes=party_votes["조"],
        poll_weight_pct=poll_weight_pct,
        party_weight_pct=party_weight_pct,
    )

    r1, r2, r3 = st.columns(3)
    r1.info(f"조가 김을 넘기기 위한 이론상 최소 확보표: {jo_needed_to_beat_kim:,.0f}표")
    r2.info(f"김이 조를 넘기기 위한 이론상 최소 확보표: {kim_needed_to_beat_jo:,.0f}표")
    r3.info(f"이가 조를 넘기기 위한 이론상 최소 확보표: {lee_needed_to_beat_jo:,.0f}표")

    st.subheader("그래프")
    g1, g2 = st.columns(2)
    with g1:
        st.pyplot(make_comparison_chart(result_df), clear_figure=True)
    with g2:
        st.pyplot(make_bar_chart(result_df, "최종점수", "최종점수 비교"), clear_figure=True)

    st.subheader("조 전략 시나리오")
    scenario_df = build_scenario_table(
        total_members=total_members,
        loss_rate_pct=loss_rate_pct,
        turnout_rate_pct=turnout_rate_pct,
        poll_weight_pct=poll_weight_pct,
        party_weight_pct=party_weight_pct,
        poll_scores=simulation["poll_scores"],
        jo_start=max(jo_start, jo_end),
        jo_end=min(jo_start, jo_end),
        jo_step=jo_step,
        kim_share_of_remaining_pct=kim_share_of_remaining_pct,
    )

    scenario_display = scenario_df.copy()
    for col in ["조 책당율(%)", "김 책당율(%)", "이 책당율(%)", "조 최종점수", "김 최종점수", "이 최종점수"]:
        scenario_display[col] = scenario_display[col].map(lambda x: round(float(x), 2))
    st.dataframe(scenario_display, use_container_width=True)

    st.line_chart(
        scenario_df.set_index("조 확보표")[["조 최종점수", "김 최종점수", "이 최종점수"]]
    )

    excel_bytes = dataframe_to_excel_bytes(display_df, scenario_display)
    st.download_button(
        label="엑셀 다운로드",
        data=excel_bytes,
        file_name="primary_election_simulator.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

except ValueError as exc:
    st.error(str(exc))

st.markdown("---")
st.markdown(
    "### 실행 방법\n"
    "1. `pip install streamlit pandas matplotlib openpyxl`\n"
    "2. `C:\\victory\\app.py` 로 저장\n"
    "3. 터미널에서 `cd C:\\victory`\n"
    "4. `py -m streamlit run app.py --server.headless true` 실행"
)