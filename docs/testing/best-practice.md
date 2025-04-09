# 測試最佳實踐

## 測試架構與組織

良好的測試架構和組織是確保測試可維護性和可讀性的關鍵。在 FastAPI 應用程序中，測試應該遵循一定的結構和命名慣例。

### 測試目錄結構

推薦的測試目錄結構應該反映應用程序的結構，使測試和被測試的代碼之間的關係清晰可見：

```
project/
├── app/
│   ├── api/
│   ├── core/
│   ├── models/
│   ├── services/
│   └── utils/
├── tests/
│   ├── api/
│   ├── core/
│   ├── models/
│   ├── services/
│   ├── utils/
│   └── conftest.py
└── pytest.ini
```

### 測試命名慣例

測試文件和函數的命名應該清晰地表明它們測試的內容：

```python
# 測試文件命名
test_user_router.py  # 測試 user_router.py
test_auth_service.py  # 測試 auth_service.py

# 測試函數命名
def test_create_user_success():  # 測試成功創建用戶
def test_create_user_duplicate_email():  # 測試創建具有重複電子郵件的用戶
def test_get_user_not_found():  # 測試獲取不存在的用戶
```

## 測試隔離

每個測試應該是獨立的，不依賴於其他測試的執行順序或結果。這確保了測試可以單獨運行，並且失敗的測試不會影響其他測試。

### 使用夾具進行設置和清理

```python
# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base, get_db
from app.main import app

@pytest.fixture(scope="function")
def db_session():
    # 創建內存數據庫
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # 創建表
    Base.metadata.create_all(bind=engine)
    
    # 創建會話
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        # 清理
        db.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def client(db_session):
    # 覆蓋依賴項
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as client:
        yield client
    
    # 清理依賴覆蓋
    app.dependency_overrides = {}
```

### 使用事務回滾確保測試隔離

```python
@pytest.fixture(scope="function")
def transactional_db_session(db_session):
    # 開始事務
    db_session.begin_nested()
    
    yield db_session
    
    # 回滾事務
    db_session.rollback()
```

## 測試數據管理

測試數據的管理是測試的重要部分，良好的測試數據管理可以使測試更加可靠和可維護。

### 使用工廠函數創建測試數據

```python
# tests/factories.py
from app.models.user import User
from app.models.item import Item

def create_test_user(db, username="testuser", email="test@example.com", password="password"):
    """創建測試用戶"""
    user = User(username=username, email=email)
    user.set_password(password)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def create_test_item(db, name="Test Item", description="Test Description", price=9.99, owner_id=None):
    """創建測試項目"""
    item = Item(name=name, description=description, price=price, owner_id=owner_id)
    db.add(item)
    db.commit()
    db.refresh(item)
    return item
```

### 使用夾具提供測試數據

```python
# tests/conftest.py
import pytest
from tests.factories import create_test_user, create_test_item

@pytest.fixture
def test_user(db_session):
    return create_test_user(db_session)

@pytest.fixture
def test_items(db_session, test_user):
    items = []
    for i in range(3):
        items.append(create_test_item(db_session, name=f"Item {i}", owner_id=test_user.id))
    return items
```

## 測試覆蓋率

測試覆蓋率是衡量代碼被測試的程度的指標。高測試覆蓋率通常意味著更少的未發現的錯誤。

### 使用 pytest-cov 測量覆蓋率

```bash
pip install pytest-cov
```

```bash
# 運行測試並生成覆蓋率報告
pytest --cov=app tests/
```

### 設置覆蓋率閾值

```ini
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = --cov=app --cov-report=term-missing --cov-fail-under=80
```

## 參數化測試

參數化測試允許使用不同的輸入值運行相同的測試代碼，減少重複代碼。

### 使用 pytest.mark.parametrize

```python
# tests/api/test_user_validation.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest.mark.parametrize(
    "username,email,password,status_code,error_message",
    [
        ("", "test@example.com", "password", 422, "username"),  # 空用戶名
        ("testuser", "", "password", 422, "email"),  # 空電子郵件
        ("testuser", "test@example.com", "", 422, "password"),  # 空密碼
        ("te", "test@example.com", "password", 422, "username"),  # 用戶名太短
        ("testuser", "invalid-email", "password", 422, "email"),  # 無效的電子郵件
        ("testuser", "test@example.com", "pass", 422, "password"),  # 密碼太短
    ],
)
def test_create_user_validation(username, email, password, status_code, error_message):
    response = client.post(
        "/users/",
        json={"username": username, "email": email, "password": password},
    )
    assert response.status_code == status_code
    assert error_message in response.text
```

## 測試速度優化

測試速度是開發效率的重要因素。快速的測試套件可以更頻繁地運行，提供更快的反饋。

### 使用內存數據庫

```python
# tests/conftest.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

@pytest.fixture(scope="session")
def engine():
    return create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
```

### 使用測試標記選擇性運行測試

```python
# tests/api/test_slow_endpoints.py
import pytest

@pytest.mark.slow
def test_slow_operation():
    # 這是一個運行時間較長的測試
    ...

# 運行除了標記為 slow 的所有測試
# pytest -k "not slow"
```

### 並行運行測試

```bash
pip install pytest-xdist
```

```bash
# 使用 4 個並行進程運行測試
pytest -n 4
```

## 測試可讀性和可維護性

可讀性和可維護性是良好測試的關鍵特性。測試應該易於理解和維護。

### 使用描述性的斷言消息

```python
# tests/api/test_user_api.py
def test_get_user_by_id(client, test_user):
    response = client.get(f"/users/{test_user.id}")
    assert response.status_code == 200, f"Failed to get user with ID {test_user.id}"
    
    user_data = response.json()
    assert user_data["username"] == test_user.username, f"Username mismatch: {user_data['username']} != {test_user.username}"
    assert user_data["email"] == test_user.email, f"Email mismatch: {user_data['email']} != {test_user.email}"
```

### 使用輔助函數簡化測試

```python
# tests/utils/test_helpers.py
def assert_user_response(response_data, expected_user):
    """驗證用戶響應數據是否與預期用戶匹配"""
    assert response_data["id"] == expected_user.id
    assert response_data["username"] == expected_user.username
    assert response_data["email"] == expected_user.email
    assert "password" not in response_data

# 在測試中使用
def test_get_user(client, test_user):
    response = client.get(f"/users/{test_user.id}")
    assert response.status_code == 200
    assert_user_response(response.json(), test_user)
```

## 測試安全性

測試應該確保應用程序的安全性，包括身份驗證、授權和數據驗證。

### 測試身份驗證

```python
# tests/api/test_auth.py
def test_login_success(client, test_user):
    response = client.post(
        "/auth/token",
        data={"username": test_user.username, "password": "password"},
    )
    assert response.status_code == 200
    token_data = response.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"

def test_login_invalid_credentials(client, test_user):
    response = client.post(
        "/auth/token",
        data={"username": test_user.username, "password": "wrong-password"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect username or password"
```

### 測試授權

```python
# tests/api/test_protected_endpoints.py
def test_access_protected_endpoint_without_token(client):
    response = client.get("/users/me")
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"

def test_access_protected_endpoint_with_token(client, test_user):
    # 先獲取令牌
    login_response = client.post(
        "/auth/token",
        data={"username": test_user.username, "password": "password"},
    )
    token = login_response.json()["access_token"]
    
    # 使用令牌訪問受保護的端點
    response = client.get(
        "/users/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["id"] == test_user.id
```

## 測試 API 文檔

FastAPI 自動生成 API 文檔，我們應該測試這些文檔是否正確可訪問。

```python
# tests/api/test_docs.py
def test_docs_accessibility(client):
    response = client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_openapi_schema(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "paths" in schema
    assert "components" in schema
    assert "schemas" in schema["components"]
```

## 測試日誌和監控

測試應該確保應用程序的日誌和監控功能正常工作。

### 測試日誌記錄

```python
# tests/utils/test_logging.py
import logging
from unittest.mock import patch
from app.utils.logger import log_request

def test_log_request(caplog):
    # 設置日誌捕獲
    caplog.set_level(logging.INFO)
    
    # 創建模擬請求
    request = {"method": "GET", "url": "/api/users", "client": {"host": "127.0.0.1"}}
    
    # 調用日誌函數
    log_request(request)
    
    # 驗證日誌記錄
    assert "GET /api/users from 127.0.0.1" in caplog.text
```

### 測試監控指標

```python
# tests/utils/test_metrics.py
from unittest.mock import patch
from app.utils.metrics import increment_request_counter

def test_increment_request_counter():
    with patch("app.utils.metrics.prometheus_client.Counter.inc") as mock_inc:
        # 調用指標函數
        increment_request_counter("/api/users", "GET")
        
        # 驗證指標增加
        mock_inc.assert_called_once_with(1, {"path": "/api/users", "method": "GET"})
```

## 測試環境配置

不同的環境（開發、測試、生產）需要不同的配置。測試應該確保應用程序在不同環境中正確配置。

### 測試環境變量加載

```python
# tests/core/test_config.py
from unittest.mock import patch
import os
from app.core.config import Settings

def test_settings_from_env_vars():
    # 設置環境變量
    env_vars = {
        "APP_NAME": "Test App",
        "DATABASE_URL": "postgresql://test:test@localhost/test",
        "SECRET_KEY": "test_secret_key",
        "DEBUG": "False",
    }
    
    with patch.dict(os.environ, env_vars):
        # 加載設置
        settings = Settings()
        
        # 驗證設置
        assert settings.app_name == "Test App"
        assert settings.database_url == "postgresql://test:test@localhost/test"
        assert settings.secret_key == "test_secret_key"
        assert settings.debug is False
```

## 持續集成與測試

持續集成 (CI) 是確保代碼質量的重要實踐。測試應該在 CI 環境中自動運行。

### GitHub Actions 配置

```yaml
# .github/workflows/test.yml
name: Test

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
        SECRET_KEY: test_secret_key
      run: |
        pytest --cov=app --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v1
```

## 測試驅動開發 (TDD)

測試驅動開發是一種開發方法，先編寫測試，然後編寫滿足測試的代碼。

### TDD 工作流程

1. **編寫測試**：先編寫測試，描述預期行為。
2. **運行測試**：確認測試失敗。
3. **編寫代碼**：編寫最少的代碼使測試通過。
4. **重構**：改進代碼，確保測試仍然通過。
5. **重複**：繼續下一個功能。

```python
# 1. 編寫測試
# tests/services/test_user_service.py
def test_create_user(db_session):
    from app.services.user_service import create_user
    
    # 準備測試數據
    user_data = {
        "username": "newuser",
        "email": "newuser@example.com",
        "password": "password123"
    }
    
    # 調用被測試的函數
    user = create_user(db_session, user_data)
    
    # 驗證結果
    assert user.id is not None
    assert user.username == "newuser"
    assert user.email == "newuser@example.com"
    assert user.verify_password("password123")

# 2. 運行測試 (會失敗)
# pytest tests/services/test_user_service.py

# 3. 編寫代碼
# app/services/user_service.py
from app.models.user import User

def create_user(db, user_data):
    user = User(
        username=user_data["username"],
        email=user_data["email"]
    )
    user.set_password(user_data["password"])
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return user

# 4. 重構 (如果需要)
# 5. 重複
```

## 測試策略

一個全面的測試策略應該包括不同類型的測試，每種類型專注於不同的方面。

### 測試金字塔

測試金字塔是一種測試策略，建議大量的單元測試，適量的集成測試，少量的端到端測試。

```
      /\
     /  \
    /    \      端到端測試
   /      \
  /--------\
 /          \    集成測試
/            \
--------------
|            |    單元測試
|            |
--------------
```

### 單元測試

單元測試專注於測試單個組件或函數的行為。

```python
# tests/utils/test_password.py
from app.utils.password import hash_password, verify_password

def test_hash_password():
    password = "secure_password"
    hashed = hash_password(password)
    
    # 驗證哈希不等於原始密碼
    assert hashed != password
    
    # 驗證哈希格式正確
    assert hashed.startswith("$2b$")

def test_verify_password():
    password = "secure_password"
    hashed = hash_password(password)
    
    # 驗證正確密碼
    assert verify_password(password, hashed) is True
    
    # 驗證錯誤密碼
    assert verify_password("wrong_password", hashed) is False
```

### 集成測試

集成測試專注於測試多個組件一起工作的行為。

```python
# tests/api/test_user_creation_flow.py
def test_user_creation_flow(client, db_session):
    # 1. 創建用戶
    user_data = {
        "username": "flowuser",
        "email": "flowuser@example.com",
        "password": "password123"
    }
    response = client.post("/users/", json=user_data)
    assert response.status_code == 201
    user_id = response.json()["id"]
    
    # 2. 登錄
    login_response = client.post(
        "/auth/token",
        data={"username": "flowuser", "password": "password123"}
    )
    assert login_response.status_code == 200
    token = login_response.json()["access_token"]
    
    # 3. 獲取用戶信息
    me_response = client.get(
        "/users/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert me_response.status_code == 200
    assert me_response.json()["username"] == "flowuser"
```

### 端到端測試

端到端測試專注於測試整個系統的行為，從用戶界面到數據庫。

```python
# tests/e2e/test_user_journey.py
import pytest
from playwright.sync_api import Page

@pytest.mark.e2e
def test_user_registration_and_login(page: Page):
    # 訪問註冊頁面
    page.goto("http://localhost:8000/register")
    
    # 填寫註冊表單
    page.fill("input[name=username]", "e2euser")
    page.fill("input[name=email]", "e2euser@example.com")
    page.fill("input[name=password]", "password123")
    page.click("button[type=submit]")
    
    # 驗證成功訊息
    assert page.inner_text(".success-message") == "Registration successful"
    
    # 訪問登錄頁面
    page.goto("http://localhost:8000/login")
    
    # 填寫登錄表單
    page.fill("input[name=username]", "e2euser")
    page.fill("input[name=password]", "password123")
    page.click("button[type=submit]")
    
    # 驗證登錄成功
    assert page.inner_text(".user-info") == "Welcome, e2euser"
```

## 測試非功能需求

除了功能需求外，還應該測試非功能需求，如性能、安全性和可用性。

### 性能測試

```python
# tests/performance/test_api_performance.py
import time
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest.mark.performance
def test_api_response_time():
    # 測量 API 響應時間
    start_time = time.time()
    response = client.get("/users/")
    end_time = time.time()
    
    # 驗證響應時間在可接受範圍內
    response_time = end_time - start_time
    assert response_time < 0.1, f"API response time too slow: {response_time:.2f} seconds"
```

### 負載測試

負載測試需要專門的工具，如 Locust 或 JMeter。這裡是一個簡單的 Locust 測試示例：

```python
# locustfile.py
from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def get_users(self):
        self.client.get("/users/")
    
    @task
    def get_items(self):
        self.client.get("/items/")
    
    @task
    def login(self):
        self.client.post(
            "/auth/token",
            data={"username": "testuser", "password": "password"}
        )
```

## 測試文檔和報告

測試文檔和報告是確保測試可理解和可追蹤的重要部分。

### 使用 pytest-html 生成報告

```bash
pip install pytest-html
```

```bash
# 生成 HTML 報告
pytest --html=report.html --self-contained-html
```

### 使用 doctest 測試文檔

```python
# app/utils/math_utils.py
def add(a, b):
    """
    將兩個數字相加並返回結果。
    
    >>> add(1, 2)
    3
    >>> add(-1, 1)
    0
    >>> add(0, 0)
    0
    """
    return a + b
```

```bash
# 運行 doctest
pytest --doctest-modules app/utils/math_utils.py
```

## 總結

測試是軟件開發的重要部分，良好的測試實踐可以提高代碼質量，減少錯誤，並使代碼更易於維護。在 FastAPI 應用程序中，測試應該涵蓋 API 端點、服務層、數據訪問層和工具函數。

關鍵點：

1. **測試架構與組織**：遵循一致的目錄結構和命名慣例。
2. **測試隔離**：確保每個測試是獨立的，不依賴於其他測試。
3. **測試數據管理**：使用工廠函數和夾具管理測試數據。
4. **測試覆蓋率**：追蹤並提高測試覆蓋率。
5. **參數化測試**：使用參數化測試減少重複代碼。
6. **測試速度優化**：使用內存數據庫和並行運行測試提高速度。
7. **測試可讀性和可維護性**：使用描述性的斷言消息和輔助函數。
8. **測試安全性**：測試身份驗證、授權和數據驗證。
9. **持續集成與測試**：在 CI 環境中自動運行測試。
10. **測試驅動開發**：考慮使用 TDD 方法開發新功能。
11. **測試策略**：實施測試金字塔，包括單元測試、集成測試和端到端測試。
12. **測試非功能需求**：測試性能、安全性和可用性。
13. **測試文檔和報告**：生成測試報告並使用 doctest 測試文檔。

通過遵循這些最佳實踐，你可以建立一個健壯的測試套件，確保 FastAPI 應用程序的質量和穩定性。