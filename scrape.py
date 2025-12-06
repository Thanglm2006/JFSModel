import time
import pandas as pd
import random
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


class JobCrawler:
    def __init__(self):
        options = Options()

        # --- BẬT CHẾ ĐỘ CHẠY NGẦM (HEADLESS) TẠI ĐÂY ---
        options.add_argument("--headless=new")  # Chế độ không hiện cửa sổ (bản mới nhất)

        # Quan trọng: Cần set kích thước giả lập, nếu không web sẽ tưởng bạn đang dùng điện thoại và đổi giao diện
        options.add_argument("--window-size=1920,1080")

        # Các cấu hình chống phát hiện bot cũ
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")

        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        self.data = []

    def crawl(self, base_url_pattern, job_card_selector, title_selector, link_selector, label, start_page=1,
              max_pages=1):

        for page in range(start_page, start_page + max_pages):
            # --- TỰ TẠO URL MỚI ---
            if page == 1:
                # Trang 1 cấu trúc thường khác một chút (không có chữ trang-1)
                current_url = f"{base_url_pattern}-vi.html"
            else:
                # Trang 2 trở đi: thêm -trang-N
                current_url = f"{base_url_pattern}-trang-{page}-vi.html"

            print(f"🔄 Đang truy cập Trang {page}: {current_url}")

            try:
                self.driver.get(current_url)
                # Chỉ cần chờ load, không cần cuộn tìm nút Next nữa
                time.sleep(3)
            except:
                print(f"❌ Lỗi truy cập URL: {current_url}")
                continue

            # --- TỪ ĐÂY TRỞ XUỐNG LÀ LOGIC CÀO JOB NHƯ CŨ ---
            try:
                jobs = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, job_card_selector))
                )
            except:
                print("❌ Không tìm thấy job nào hoặc đã hết trang.")
                break

            job_links = []
            for job in jobs:
                try:
                    title = job.find_element(By.CSS_SELECTOR, title_selector).text
                    link = job.find_element(By.CSS_SELECTOR, link_selector).get_attribute('href')
                    if link:
                        job_links.append((title, link))
                except:
                    continue

            print(f"🔎 Tìm thấy {len(job_links)} công việc. Bắt đầu lấy nội dung...")

            for title, link in job_links:
                try:
                    self.driver.execute_script(f"window.open('{link}', '_blank');")
                    self.driver.switch_to.window(self.driver.window_handles[-1])
                    time.sleep(random.uniform(1, 2))

                    description = "Không lấy được nội dung"
                    try:
                        detail_elements = self.driver.find_elements(By.CSS_SELECTOR, ".detail-row")
                        if detail_elements:
                            description = "\n".join([elem.text for elem in detail_elements])
                        else:
                            description = self.driver.find_element(By.TAG_NAME, "body").text
                    except:
                        pass

                    if len(description) > 50:
                        self.data.append({
                            'title': title,
                            'description': description,
                            'label': label,
                            'source': link
                        })

                    self.driver.close()
                    self.driver.switch_to.window(self.driver.window_handles[0])

                except Exception as e:
                    print(f"⚠️ Lỗi job: {e}")
                    if len(self.driver.window_handles) > 1:
                        self.driver.close()
                    self.driver.switch_to.window(self.driver.window_handles[0])

            # --- KHÔNG CẦN LOGIC CLICK NÚT NEXT NỮA ---
            print(f"✅ Xong trang {page}.")
    def save_csv(self, filename="raw_jobs.csv"):
        if not self.data:
            print("⚠️ Không có dữ liệu nào được cào! Vui lòng kiểm tra lại CSS Selector hoặc đường truyền.")
            return
        df = pd.DataFrame(self.data)
        # Làm sạch cơ bản: Xóa xuống dòng thừa
        df['description'] = df['description'].apply(lambda x: x.replace('\n', ' ').strip())
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✅ Đã lưu {len(df)} dòng vào file {filename}")
        self.driver.quit()


# --- HƯỚNG DẪN SỬ DỤNG ---

crawler = JobCrawler()

# VÍ DỤ 1: Cào CareerViet (Nguồn uy tín - Label 1)
# URL: https://careerviet.vn/viec-lam/tat-ca-viec-lam-vi.html
# Cách lấy CSS Selector: Chuột phải vào Tiêu đề job -> Inspect (Kiểm tra) -> Xem class
print("🕷️ Bắt đầu cào CareerViet...")
crawler.crawl(
    # Lưu ý: Cắt bỏ "-vi.html" ở cuối, chỉ để lại phần gốc
    base_url_pattern="https://careerviet.vn/viec-lam/tat-ca-viec-lam",

    job_card_selector=".job-item",
    title_selector=".title a",
    link_selector=".title a",
    label=1,
    start_page=1,  # Bắt đầu từ trang 1
    max_pages=50  # Cào 5 trang (Trang 1 -> Trang 5)
)
# # VÍ DỤ 2: Cào Muaban.net (Nguồn hỗn hợp/tiềm năng lừa đảo - Label 0)
# # Lưu ý: Muaban.net có cả việc thật, bạn cào xong phải lọc tay lại những bài lừa đảo để gán Label 0
# print("🕷️ Bắt đầu cào Muaban.net...")
# crawler.crawl(
#     url="https://muaban.net/viec-lam-tuyen-dung-toan-quoc-l0-c100",
#     job_card_selector=".list-item-container",  # Cần F12 để check lại class này tùy thời điểm
#     title_selector=".title",
#     link_selector=".title a",
#     label=0  # Tạm gán là 0, sau này lọc lại
# )

crawler.save_csv("data_viet.csv")