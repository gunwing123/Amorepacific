import os
import asyncio
import threading
import sys
import subprocess
from pathlib import Path
import schedule
import time

# 작업 디렉토리 설정 (필요시 수정)
os.chdir(r'C:/Users/kyj20/Desktop/용준/공모전/아모레퍼시픽/AI agent')

# Playwright 브라우저 설치 확인 및 자동 설치
def ensure_playwright_browsers():
    """Playwright 브라우저가 설치되어 있는지 확인하고, 없으면 자동 설치"""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            # chromium이 설치되어 있는지 확인
            try:
                browser = p.chromium.launch(headless=True)
                browser.close()
                print("Playwright 브라우저 확인 완료")
                return
            except Exception:
                pass
    except Exception:
        pass
    
    # 브라우저가 없으면 설치
    print("Playwright 브라우저를 설치합니다... (처음 실행 시 시간이 걸릴 수 있습니다)")
    try:
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], 
                      check=True, capture_output=True, text=True)
        print("Playwright 브라우저 설치 완료!")
    except subprocess.CalledProcessError as e:
        print(f"브라우저 설치 중 오류 발생: {e}")
        print("수동으로 설치하려면 터미널에서 다음 명령을 실행하세요:")
        print("  python -m playwright install chromium")
        raise

# 브라우저 설치 확인
#ensure_playwright_browsers()

# main.py의 함수들을 직접 호출
from main import collect, collect_brand_products, run_analysis



def run_in_thread(coro):
    """별도 스레드에서 새로운 이벤트 루프로 실행 (Windows 서브프로세스 지원)"""
    result = [None]
    exception = [None]
    
    def run():
        try:
            # Windows에서 서브프로세스를 지원하는 ProactorEventLoop 사용
            if sys.platform == 'win32':
                loop = asyncio.ProactorEventLoop()
            else:
                loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result[0] = loop.run_until_complete(coro)
        except Exception as e:
            exception[0] = e
        finally:
            if 'loop' in locals():
                loop.close()
    
    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    
    if exception[0]:
        raise exception[0]
    return result[0]

def starts():
    # 옵션 1: 전체 랭킹 수집
    print("랭킹 수집 시작...")
    run_in_thread(collect(save_html=True, show_sample=True, wait_ms=4000))
    
    # 옵션 2: 브랜드 제품 상세 수집 (추천)
    print("브랜드 제품 상세 수집 시작...")
    run_in_thread(collect_brand_products(save_html=True, show_sample=True, wait_ms=4000))
    
    # 옵션 3: 인사이트 생성 (수집 후 실행)
    #print("인사이트 생성 시작...")
    #run_analysis()

#schedule.every().friday.at("12:30").do(starts)

# 옵션 1: 전체 랭킹 수집
print("랭킹 수집 시작...")
run_in_thread(collect(save_html=True, show_sample=True, wait_ms=4000))

# 옵션 2: 브랜드 제품 상세 수집 (추천)
print("브랜드 제품 상세 수집 시작...")
run_in_thread(collect_brand_products(save_html=True, show_sample=True, wait_ms=4000))

# 옵션 3: 인사이트 생성 (수집 후 실행)
print("인사이트 생성 시작...")
run_analysis()


