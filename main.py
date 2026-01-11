"""
@COSME 라네즈 랭킹 수집/분석 스크립트.

- 매일 1회 실행해 CSV에 기록
- 전체 카테고리 기준
- 알림 채널 없음, CSV 출력만
"""

import re
import argparse
import asyncio
import csv
import os
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

# 수집 대상 설정
CATEGORY_URLS: Dict[str, str] = {
    # @COSME 전체 랭킹 (PC)
    "All": "https://www.cosme.net/ranking/products/page/",
    "Skin-rise" : "https://www.cosme.net/categories/item/800/ranking-rise/",
    "Facial-rise" : "https://www.cosme.net/categories/item/1005/ranking-rise/",
    "Facialmask-rise" : "https://www.cosme.net/categories/item/904/ranking-rise/",
    "Basemakeup-rise" : "https://www.cosme.net/categories/item/804/ranking-rise/"
}
#파일 데이터
file_name = ["_All", "_Skin_rise","_Facial_rise","_Facialmask_rise","_Basemakeup_rise"]

# 라네즈 브랜드 제품 리스트 페이지 (개별 제품 페이지를 모두 눌러 수집)
BRAND_PRODUCT_LIST_URL = "https://www.cosme.net/brands/7623/product/?page="

# 라네즈 브랜드 키워드
BRAND_KEYWORDS = ["LANEIGE", "ラネージュ"]

# 출력 경로
DATA_DIR = Path("data")
HISTORY_CSV = DATA_DIR / "rank_history.csv"
DAILY_DIR = DATA_DIR / "daily"
DEBUG_DIR = DATA_DIR / "debug"
REPORTS_DIR = Path("reports")
PRODUCT_CSV = DATA_DIR / "brand_products.csv"
PRODUCT_DAILY_DIR = DATA_DIR / "daily_products"


@dataclass
class RankItem:#랭킹 페이지 표현
    scraped_date: date
    category: str
    rank: int
    product_name: str
    product_category : str
    product_star : float
    reviews_text : int
    product_launch_day : str

@dataclass
class ProductItem:#개별 제품 페이지
    scraped_date: date
    product_name: str
    product_url: str
    rank_value: Optional[int]
    rank_text: Optional[str]
    release_date: Optional[str]
    category_text: Optional[str]
    image_url: Optional[str] = None
    rating_text: Optional[str] = None
    reviews_text: Optional[str] = None


def ensure_dirs() -> None:#폴더 생성
    DATA_DIR.mkdir(exist_ok=True)
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    PRODUCT_DAILY_DIR.mkdir(parents=True, exist_ok=True)


async def fetch_html(url: str, wait_ms: int = 2000) -> str:
    #url을 통한 HTML반환
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle", timeout=60_000)
        await page.wait_for_timeout(wait_ms)
        html = await page.content()
        await browser.close()
        return html

#수집단계
async def collect(save_html: bool = False, show_sample: bool = False, wait_ms: int = 2000) -> None:
    """전체 카테고리 1회 수집."""
    today = date.today()
    all_records: List[RankItem] = []
    for category, url_main in CATEGORY_URLS.items():
        if 'All' in category:
            for i in range(10):
                url = url_main + str(i)
                html = await fetch_html(url, wait_ms=wait_ms)
                if save_html:
                    ensure_dirs()
                    (DEBUG_DIR / f"{today.isoformat()}_{category}.html").write_text(html, encoding="utf-8-sig")
                records = parse_rankings(category, html, today, category,i*10)
                append_history(records)
                all_records.extend(records)
                if show_sample:
                    sample = records[:10]
                    print(f"[DEBUG] {category} {i*10+1}~{i*10+10} 추출 샘플 {len(sample)}개 / 총 {len(records)}개")
                    for r in sample:
                        print(f"  #{r.rank} {r.product_name}")
        
        else:
            html = await fetch_html(url, wait_ms=wait_ms)
            if save_html:
                ensure_dirs()
                (DEBUG_DIR / f"{today.isoformat()}_{category}.html").write_text(html, encoding="utf-8-sig")
            records = parse_rankings(category, html, today, category,i*10)
            append_history(records)
            all_records.extend(records)
            if show_sample:
                sample = records[:10]
                print(f"[DEBUG] {category} 1~10 추출 샘플 {len(sample)}개 / 총 {len(records)}개")
                for r in sample:
                    print(f"  #{r.rank} {r.product_name}")
    
    print(f"수집 완료: {len(all_records)}건")

def parse_rankings(cat : str, html: str, scraped_date: date, category: str, rank) -> List[RankItem]:
    """HTML에서 랭킹 아이템 파싱 (유연한 셀렉터 사용)."""
    soup = BeautifulSoup(html, "html.parser")
    items: List[RankItem] = []
    # all 전체 요소 가져오기
    # brand, name, genre, star,lauch day,shop
    if 'all' in cat:
        parents = soup.select('#list-item .summary ')
    else:
        parents = soup.select('#keyword-ranking-list .summary')
    for child in parents:
        texts = [el.get_text(strip=True) for el in child if el.get_text(strip=True)]
        rank+=1
        texts.append(rank)
        if BRAND_KEYWORDS[0] in texts[0]:
            items.append(
                RankItem(
                    scraped_date=scraped_date,
                    category=category,
                    rank=texts[-1],
                    product_name=texts[1],
                    product_category = texts[2],
                    product_star = re.findall(r"\d+\.\d+",texts[3])[0],
                    reviews_text=re.findall(r"(\d+)件",texts[3])[0],
                    product_launch_day = re.findall(r"\d{4}/\d{1,2}/\d{1,2}", texts[4])[0]
                    )
                )
    return items
    

#엑셀 제작 단계
def append_history(records: List[RankItem]) -> None:
    """CSV에 기록 추가."""
    if not records:
        return
    ensure_dirs()
    headers = list(asdict(records[0]).keys())
    # 전체 히스토리 파일은 항상 append
    file_exists = HISTORY_CSV.exists()
    with HISTORY_CSV.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

    # 일자+카테고리별 파일도 여러 페이지를 합쳐서 보관해야 하므로 append 모드 사용
    daily_path = DAILY_DIR / f"{records[0].scraped_date.isoformat()}_{records[0].category}.csv"
    daily_exists = daily_path.exists()
    with daily_path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not daily_exists:
            writer.writeheader()

        for r in records:
            writer.writerow(asdict(r))


def append_product_history(records: List[ProductItem]) -> None:
    """브랜드 제품 상세 수집 결과 기록."""
    if not records:
        return
    ensure_dirs()
    headers = list(asdict(records[0]).keys())
    file_exists = PRODUCT_CSV.exists()
    with PRODUCT_CSV.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

    daily_path = PRODUCT_DAILY_DIR / f"{records[0].scraped_date.isoformat()}_brand_products.csv"
    with daily_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


#브랜드에서 제품명, url 추출
def parse_product_list(html: str) -> List[Tuple[str, str]]:
    
    soup = BeautifulSoup(html, "html.parser")
    results: List[Tuple[str, str]] = []
    
    for i in range(1,21):
        parents = soup.select(f'#productListLarge > div:nth-child({i}) > div > div > div > div.productName > div.brand-renewal-product-area > h4 > a')
        if not parents:
            if i == 1:
                return None
            break
        for child in parents:
            link = child.get("href")
            name = child.get_text(strip=True)
            results.append((name, link))
        
    
    # 중복 제거 (URL 기준)
    seen_urls = set()
    unique = []
    for name, href in results:
        if href in seen_urls:
            continue
        seen_urls.add(href)
        unique.append((name, href))
    return unique


#제품 사이트 세부사항 추출
def parse_product_detail(html: str, url: str, scraped_date: date) -> ProductItem:
    soup = BeautifulSoup(html, "html.parser")

    # 제품명
    parents = soup.select('#product-header > h2 > strong > a')[0]
    product_name = parents.get_text(strip=True)
    
    #랭크
    rankwname = soup.select('#pdt-info-newdb-1606 > div.info > div > ul > li.info-rank.clearfix > p.info-desc.clearfix')
    if rankwname:
        rankwname = rankwname[0]
        rankwname = rankwname.get_text()
        m = re.match(r"^(\d+)\s*位?(.*)$", rankwname)
        rank_value, rank_text = m.group(1), m.group(2)
    else:
        rank_value, rank_text = '-', '-'
    
    #발매일
    if '-' not in rank_value:
        m = soup.select('#pdt-info-newdb-1606 > div.info > div > ul > li:nth-child(4) > p.info-desc')
        m = m[0]
        release_date = m.get_text().strip()
    else:
        m = soup.select('#pdt-info-newdb-1606 > div.info > div > ul > li:nth-child(3) > p.info-desc')
        m = m[0]
        release_date = m.get_text().strip()
    
    #카테고리
    m = soup.select('#product-spec > dl.item-category.clearfix > dd > span')
    if m:
        m = m[0]
        m = m.get_text().split("\xa0>\xa0")
        category_text = m[-1]
    else:
        category_text = '-'
    
    #제품 이미지
    m = soup.select('#main > div.vri-item > div.vri-item-inr-top > ul > li:nth-child(1) > a > p.vari-pic > img ')
    if m:
        m = m[0]
        image_url = m.get('src')
    else:
        image_url = '-'
    
    #별점
    m = soup.select_one('#pdt-info-newdb-1606 > div.info > div > ul > li.info-rev.clearfix > div > p.average')
    if m:
        rating_text = m.get_text()
    else:
        rating_text = '-'
    
    #리뷰갯수
    m = soup.select('#product-header > div.navi-tab-wrap.navi-tab-top > ul > li.review > a > span')
    if m:
        m = m[0]
        text = m.get_text()
        reviews_text = text.replace("(", "").replace(")", "")
    else:
        reviews_text = '-'


    return ProductItem(
        scraped_date=scraped_date,
        product_name=product_name,
        product_url=url,
        rank_value=rank_value,
        rank_text=rank_text,
        release_date=release_date,
        category_text=category_text,
        image_url=image_url,
        rating_text=rating_text,
        reviews_text=reviews_text
    )


#브랜드별 데이터 추출
async def collect_brand_products(save_html: bool = False, show_sample: bool = False, wait_ms: int = 2000) -> None:
    ensure_dirs()
    today = date.today()
    print(f"[1/3] 제품 리스트 페이지(1~30) 수집 중: {BRAND_PRODUCT_LIST_URL}")
    all_products: List[Tuple[str, str]] = []
    for page_idx in range(1, 31):
        page_url = f"{BRAND_PRODUCT_LIST_URL}{page_idx}"
        print(f"   - ({page_idx}/30) 요청: {page_url}")
        list_html = await fetch_html(page_url, wait_ms=wait_ms)
        debug_path = DEBUG_DIR / f"{today.isoformat()}_brand_list_p{page_idx}.html"
        debug_path.write_text(list_html, encoding="utf-8-sig")
        parsed = parse_product_list(list_html)
        if not parsed:
            print(f"     ⚠️ {page_idx}페이지에서 제품을 찾지 못했습니다. HTML 확인: {debug_path}")
            break
        all_products.extend(parsed)

    print(f"[2/3] 완료: 총 {len(all_products)}개 제품 링크 발견")
    
    print("[3/3] 제품 상세 페이지 수집 중...")
    collected: List[ProductItem] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        for idx, (name, href) in enumerate(all_products, 1):        
            try:
                await page.goto(href, wait_until="load", timeout=90_000)
                await page.wait_for_timeout(wait_ms)
                detail_html = await page.content()
                if save_html:
                    (DEBUG_DIR / f"{today.isoformat()}_product_{idx}.html").write_text(detail_html, encoding="utf-8-sig")
                item = parse_product_detail(detail_html, href, today)
                collected.append(item)
                if show_sample and idx <= 5:
                    print(f"   [#{idx}] {item.product_name[:40]} | 랭킹={item.rank_value} | 출고일={item.release_date}")
                elif idx % 10 == 0:
                    print(f"   진행: {idx}/{len(all_products)} ({len(collected)}건 수집됨)")
            except Exception as e:
                print(f"   [WARN] {idx}/{len(all_products)} 실패: {href} -> {e}")
        await browser.close()

    append_product_history(collected)
    print(f"[3/3] 완료: 총 {len(collected)}건 수집")
    print(f"✅ 결과 저장: {PRODUCT_CSV}")

#인사이트 도출
def to_drive():
    """인사이트"""
    from openai import OpenAI
    texts = {}
    rank_folder_path = ""#전체순위
    product_folder_path = ""#제품별 순위
    today = date.today()
    one_weeks_ago = today - timedelta(weeks=1)
    
    #파일 가져오기
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2})_.+\.csv")
    for filename in os.listdir(rank_folder_path):
        match = pattern.match(filename)
        if match:
            file_date = datetime.strptime(match.group(1), "%Y-%m-%d").date()
            if one_weeks_ago <= file_date:
                file_path = os.path.join(rank_folder_path, filename)
                df = pd.read_csv(file_path)
                texts[filename] = df.to_string(index=False)
    
    for filename in os.listdir(product_folder_path):
        match = pattern.match(filename)
        if match:
            file_date = datetime.strptime(match.group(1), "%Y-%m-%d").date()
            if one_weeks_ago <= file_date and file_date <= today:
                file_path = os.path.join(product_folder_path, filename)
                df = pd.read_csv(file_path)
                df.drop(['product_url','image_url'], axis = 1, inplace = True)
                texts[filename] = df.to_string(index=False)
    
                
    """AI인사이트 도출"""
    client = OpenAI(api_key="")

    prompt = f"""
    다음은 엑셀 파일 데이터입니다.
    
    [제품 리스트]
    {texts}
    딕셔너리 형식으로 되어있는데 
    만약 all 하고 rise가 없으면 '전체 랭킹 및 급상승 랭킹에 없음' 인사이트 적어주고, Brand_product 데이터만 이용해서 일자별 추세 분석해줘
    공통 : product 파일의 과거 데이터부터 현재 데이터까지 연결해서 "~제품이 ~주간 ~위"형식으로 인사이트 제공해줘
    1. All 과 product 파일에 공통된 제품 이름이 있으면 "~제품이 ~주간 전체 순위 ~위, 카테고리 ~위"
    2. Rise 와 product 파일에 공통된 제품 이름이 있으면 "~제품이 발매일 ~동안 ~부분에서 카테고리 급상승 순위 ~위"
    일본어는 한국어로 변경해서 작성해주세요
    인사이트 분석 이후 리뷰 수·평점(rating_text, reviews_text) 변화까지 포함한 시계열 분석(증감률, 그래프용 데이터) 제공
    """
    
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    
    print("LLM 인사이트 결과:")
    insight = response.choices[0].message.content
    with open(f'', 'w', encoding='utf-8') as f:
        print(insight, file = f)
    print(insight)





    
def run_analysis() -> None:
    to_drive()


def main() -> None:
    parser = argparse.ArgumentParser(description="COSME 랭킹 수집/분석")
    sub = parser.add_subparsers(dest="command")
    collect_parser = sub.add_parser("collect", help="랭킹 수집 후 CSV 기록")
    collect_parser.add_argument("--save-html", action="store_true", help="HTML 원본을 data/debug에 저장")
    collect_parser.add_argument("--show-sample", action="store_true", help="파싱된 상위 5개를 콘솔에 표시")
    collect_parser.add_argument("--wait-ms", type=int, default=2000, help="페이지 로딩 대기(ms)")
    product_parser = sub.add_parser("collect-products", help="브랜드 제품 상세를 모두 순회하며 수집")
    product_parser.add_argument("--save-html", action="store_true", help="제품 상세 HTML을 data/debug에 저장")
    product_parser.add_argument("--show-sample", action="store_true", help="상위 5개 제품을 콘솔에 표시")
    product_parser.add_argument("--wait-ms", type=int, default=2000, help="페이지 로딩 대기(ms)")
    sub.add_parser("analyze", help="CSV 기반 인사이트 생성")
    args = parser.parse_args()

    if args.command == "collect":
        asyncio.run(
            collect(
                save_html=args.save_html,
                show_sample=args.show_sample,
                wait_ms=args.wait_ms,
            )
        )
    elif args.command == "collect-products":
        asyncio.run(
            collect_brand_products(
                save_html=args.save_html,
                show_sample=args.show_sample,
                wait_ms=args.wait_ms,
            )
        )
    elif args.command == "analyze":
        run_analysis()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

