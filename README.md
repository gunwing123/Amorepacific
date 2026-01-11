# @COSME 라네즈 랭킹 수집/분석 Agent

- 전체 카테고리 기준으로 @COSME 랭킹에서 라네즈 제품만 필터링
- 하루 1회 실행을 전제로 CSV에 기록, 간단 인사이트 리포트 생성
- 알림 채널 없음

## 설치
```
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
python -m playwright install chromium
```

## 사용법
- 수집: `python main.py collect`
- 인사이트 생성: `python main.py analyze`
- 결과물
  - `data/rank_history.csv`: 전체 히스토리
  - `data/daily/YYYY-MM-DD_all.csv`: 일자별 스냅샷
  - `reports/insights.csv`, `reports/summary.md`: 간단 인사이트 요약

## 참고
- DOM 변경 시 `parse_rankings` 내 셀렉터를 수정하면 됨.
- 카테고리 추가 시 `CATEGORY_URLS`에 URL을 확장.

