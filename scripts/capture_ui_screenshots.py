"""
Script to launch Flask app in a background thread and use headless Selenium Chromium
to capture high-resolution screenshots of the UI under different dummy transaction scenarios.
"""

import os
import sys
import time
import threading
from wsgiref.simple_server import make_server

# Add repo root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By


def run_server(server):
    try:
        server.serve_forever()
    except Exception:
        pass


def capture_screenshots():
    output_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'artifacts', 'file_generation', 'ttl=63d', 'output'
    ))
    os.makedirs(output_dir, exist_ok=True)

    port = 5055
    server = make_server('127.0.0.1', port, app.wsgi_app)
    t = threading.Thread(target=run_server, args=(server,), daemon=True)
    t.start()
    time.sleep(1.5)  # Wait for server to start

    # Configure headless Chromium
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1440,1100')
    options.binary_location = '/usr/bin/chromium'

    service = Service('/usr/bin/chromedriver')
    driver = webdriver.Chrome(service=service, options=options)

    try:
        # 1. Capture Scenario 1 (Default: Critical Mule Drain)
        url = f'http://127.0.0.1:{port}/'
        driver.get(url)
        time.sleep(1.0)
        shot1_path = os.path.join(output_dir, 'flask_ui_scenario_1_mule_drain.png')
        driver.save_screenshot(shot1_path)
        print(f"Captured: {shot1_path}")

        # 2. Capture Scenario 2 (Legitimate Merchant Payment)
        btn_normal = driver.find_element(By.ID, 'btn-normal')
        btn_normal.click()
        time.sleep(1.2)
        shot2_path = os.path.join(output_dir, 'flask_ui_scenario_2_routine_payment.png')
        driver.save_screenshot(shot2_path)
        print(f"Captured: {shot2_path}")

        # 3. Capture Scenario 3 (Night Burst)
        btn_night = driver.find_element(By.ID, 'btn-night')
        btn_night.click()
        time.sleep(1.2)
        shot3_path = os.path.join(output_dir, 'flask_ui_scenario_3_night_burst.png')
        driver.save_screenshot(shot3_path)
        print(f"Captured: {shot3_path}")

    finally:
        driver.quit()
        server.shutdown()
        print("Completed screenshot capture.")


if __name__ == '__main__':
    capture_screenshots()
