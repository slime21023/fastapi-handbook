# 整合測試

## 整合測試的基本概念

整合測試是測試金字塔的中間層，專注於測試多個組件如何協同工作。在 FastAPI 應用程序中，整合測試主要關注 API 端點和依賴項的交互，以及與外部系統（如數據庫、緩存或外部服務）的集成。

| 概念 | 說明 |
|------|------|
| **測試範圍** | 多個組件的協同工作 |
| **隔離程度** | 部分隔離，可能包含部分真實依賴 |
| **執行速度** | 中等，通常秒級 |
| **依賴處理** | 可能使用真實依賴或測試替身 |
| **數量比例** | 在測試金字塔中佔比約 20-25% |

## FastAPI 中的整合測試目標

在 FastAPI 應用中，以下是整合測試的主要目標：

| 測試目標 | 測試重點 | 示例 |
|---------|---------|------|
| **API 端點** | 路由處理、請求/響應流程 | 測試 GET/POST/PUT/DELETE 端點 |
| **中間件** | 請求處理流程中的中間件功能 | 認證、日誌、CORS 處理 |
| **依賴注入鏈** | 多個依賴項的協同工作 | 認證+授權+資源獲取流程 |
| **數據庫交互** | ORM/查詢操作與業務邏輯的集成 | 創建/讀取/更新/刪除操作 |
| **外部服務集成** | 與外部 API 或服務的交互 | 支付處理、郵件發送、文件存儲 |

## FastAPI TestClient 簡介

FastAPI 提供了 `TestClient` 類，這是整合測試的核心工具，它基於 `httpx` 庫，允許你向應用程序發送請求並檢查響應，而無需實際啟動服務器。

| 特性 | 說明 |
|------|------|
| **無服務器測試** | 直接測試 ASGI 應用，無需啟動實際服務器 |
| **完整請求流程** | 模擬完整的 HTTP 請求/響應周期 |
| **中間件支持** | 測試包括中間件在內的完整應用堆棧 |
| **會話支持** | 維護 cookie 和會話狀態 |
| **同步 API** | 提供同步接口，簡化測試編寫 |

### 基本用法示例

```python
from fastapi.testclient import TestClient
from app.main import app  # 你的 FastAPI 應用

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}
```

## 設置整合測試環境

### 測試配置與夾具

使用 pytest 夾具 (fixtures) 可以有效管理測試環境的設置和清理：

```python
# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import Base, get_db

# 創建測試數據庫引擎
TEST_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(TEST_DATABASE_URL)

# 創建測試會話
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")
def db():
    # 創建表
    Base.metadata.create_all(bind=engine)
    
    # 創建會話
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        
    # 清理表
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def client(db):
    # 覆蓋依賴
    def override_get_db():
        try:
            yield db
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as c:
        yield c
    
    # 清理依賴覆蓋
    app.dependency_overrides = {}
```

## API 端點整合測試

### 測試策略

| 策略 | 說明 |
|------|------|
| **CRUD 操作測試** | 測試創建、讀取、更新和刪除資源的端點 |
| **認證/授權測試** | 測試需要認證的端點和權限檢查 |
| **錯誤處理測試** | 測試端點對錯誤輸入的處理 |
| **業務流程測試** | 測試涉及多個端點的業務流程 |

### 示例：簡單商品 API 端點測試

假設我們有以下商品 API 端點：

```python
# app/routers/items.py
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from app.schemas import Item, ItemCreate
from app.services.item_service import ItemService
from app.dependencies.database import get_item_service

router = APIRouter(prefix="/items", tags=["items"])

@router.post("/", response_model=Item, status_code=status.HTTP_201_CREATED)
def create_item(item: ItemCreate, service: ItemService = Depends(get_item_service)):
    return service.create_item(item)

@router.get("/", response_model=List[Item])
def read_items(skip: int = 0, limit: int = 100, service: ItemService = Depends(get_item_service)):
    return service.get_items(skip=skip, limit=limit)

@router.get("/{item_id}", response_model=Item)
def read_item(item_id: int, service: ItemService = Depends(get_item_service)):
    item = service.get_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="商品不存在")
    return item
```

對應的整合測試：

```python
# tests/api/test_items.py
import pytest

def test_create_item(client):
    # Arrange
    item_data = {
        "name": "測試商品",
        "description": "這是一個測試商品",
        "price": 99.99
    }
    
    # Act
    response = client.post("/items/", json=item_data)
    
    # Assert
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == item_data["name"]
    assert data["price"] == item_data["price"]
    assert "id" in data

def test_read_items(client):
    # Arrange - 創建幾個商品
    client.post("/items/", json={"name": "商品1", "price": 10.0})
    client.post("/items/", json={"name": "商品2", "price": 20.0})
    
    # Act
    response = client.get("/items/")
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["name"] == "商品1"
    assert data[1]["name"] == "商品2"

def test_read_item(client):
    # Arrange - 創建一個商品
    create_response = client.post("/items/", json={"name": "測試商品", "price": 15.0})
    item_id = create_response.json()["id"]
    
    # Act
    response = client.get(f"/items/{item_id}")
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == item_id
    assert data["name"] == "測試商品"
    assert data["price"] == 15.0

def test_read_item_not_found(client):
    # Act
    response = client.get("/items/999")  # 假設 ID 999 不存在
    
    # Assert
    assert response.status_code == 404
    assert "商品不存在" in response.json()["detail"]
```

## 認證和授權的整合測試

測試需要認證的 API 端點是整合測試的重要部分。

### 測試策略

| 策略 | 說明 |
|------|------|
| **模擬認證** | 創建測試用戶和令牌 |
| **權限測試** | 測試不同角色的用戶對受保護資源的訪問 |
| **令牌失效測試** | 測試過期或無效令牌的處理 |
| **登錄流程測試** | 測試完整的登錄/登出流程 |

### 示例：簡單認證 API 測試

假設我們有一個簡單的登錄端點：

```python
# app/routers/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from app.services.auth_service import AuthService
from app.schemas import Token
from app.dependencies.services import get_auth_service

router = APIRouter(tags=["authentication"])

@router.post("/token", response_model=Token)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    auth_service: AuthService = Depends(get_auth_service)
):
    user = auth_service.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用戶名或密碼錯誤",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = auth_service.create_access_token(user_id=user.id)
    return {"access_token": access_token, "token_type": "bearer"}
```

對應的整合測試：

```python
# tests/api/test_auth.py
import pytest

@pytest.fixture
def test_user(client):
    # 創建測試用戶
    user_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "password123"
    }
    response = client.post("/users/", json=user_data)
    return response.json()

def test_login_success(client, test_user):
    # Arrange
    login_data = {
        "username": "testuser",
        "password": "password123"
    }
    
    # Act
    response = client.post(
        "/token",
        data=login_data,  # 注意：登錄端點期望表單數據，而不是 JSON
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

def test_login_invalid_credentials(client, test_user):
    # Arrange
    login_data = {
        "username": "testuser",
        "password": "wrongpassword"  # 錯誤的密碼
    }
    
    # Act
    response = client.post(
        "/token",
        data=login_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    
    # Assert
    assert response.status_code == 401
    assert "用戶名或密碼錯誤" in response.json()["detail"]

def test_protected_route(client, test_user):
    # Arrange - 先登錄獲取令牌
    login_response = client.post(
        "/token",
        data={"username": "testuser", "password": "password123"},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    token = login_response.json()["access_token"]
    
    # Act - 訪問受保護的路由
    response = client.get(
        "/users/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "testuser"
```

## 數據庫整合測試

測試與數據庫交互的代碼是整合測試的重要部分。

### 測試策略

| 策略 | 說明 |
|------|------|
| **測試數據庫** | 使用專用的測試數據庫或內存數據庫 |
| **事務回滾** | 使用事務確保測試之間的隔離 |
| **數據填充** | 創建測試所需的初始數據 |
| **ORM 操作測試** | 測試數據庫操作與業務邏輯的集成 |

### 示例：簡單商品存儲庫測試

假設我們有一個簡單的商品存儲庫：

```python
# app/repositories/item_repository.py
from sqlalchemy.orm import Session
from typing import List, Optional

from app.models.item import Item
from app.schemas.item import ItemCreate

class ItemRepository:
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, item_data: ItemCreate) -> Item:
        """創建新商品"""
        db_item = Item(**item_data.dict())
        self.db.add(db_item)
        self.db.commit()
        self.db.refresh(db_item)
        return db_item
    
    def get_by_id(self, item_id: int) -> Optional[Item]:
        """根據 ID 獲取商品"""
        return self.db.query(Item).filter(Item.id == item_id).first()
    
    def get_all(self, skip: int = 0, limit: int = 100) -> List[Item]:
        """獲取所有商品"""
        return self.db.query(Item).offset(skip).limit(limit).all()
```

對應的整合測試：

```python
# tests/repositories/test_item_repository.py
import pytest
from app.repositories.item_repository import ItemRepository
from app.schemas.item import ItemCreate

@pytest.fixture
def item_repo(db):
    return ItemRepository(db)

def test_create_item(item_repo):
    # Arrange
    item_data = ItemCreate(name="測試商品", price=99.99)
    
    # Act
    item = item_repo.create(item_data)
    
    # Assert
    assert item.id is not None
    assert item.name == "測試商品"
    assert item.price == 99.99

def test_get_item_by_id(item_repo):
    # Arrange
    item_data = ItemCreate(name="測試商品", price=99.99)
    created_item = item_repo.create(item_data)
    
    # Act
    item = item_repo.get_by_id(created_item.id)
    
    # Assert
    assert item is not None
    assert item.id == created_item.id
    assert item.name == "測試商品"

def test_get_all_items(item_repo):
    # Arrange
    item_repo.create(ItemCreate(name="商品1", price=10.0))
    item_repo.create(ItemCreate(name="商品2", price=20.0))
    
    # Act
    items = item_repo.get_all()
    
    # Assert
    assert len(items) == 2
    assert items[0].name == "商品1"
    assert items[1].name == "商品2"
```

## 模擬外部服務

在整合測試中，我們可能需要模擬外部服務，如支付處理、電子郵件發送或第三方 API。

### 測試策略

| 策略 | 說明 |
|------|------|
| **服務模擬** | 創建外部服務的模擬版本 |
| **依賴注入** | 使用依賴注入替換真實服務 |
| **響應模擬** | 模擬外部服務的各種響應情況 |
| **錯誤處理測試** | 測試外部服務失敗時的處理 |

### 示例：簡單郵件服務測試

假設我們有一個簡單的郵件服務：

```python
# app/services/email_service.py
from typing import List
import smtplib
from email.message import EmailMessage

from app.config import settings

class EmailService:
    def __init__(self, smtp_server=None, smtp_port=None, username=None, password=None):
        self.smtp_server = smtp_server or settings.SMTP_SERVER
        self.smtp_port = smtp_port or settings.SMTP_PORT
        self.username = username or settings.SMTP_USERNAME
        self.password = password or settings.SMTP_PASSWORD
    
    def send_email(self, to_email: str, subject: str, content: str) -> bool:
        """發送電子郵件"""
        msg = EmailMessage()
        msg.set_content(content)
        msg["Subject"] = subject
        msg["From"] = self.username
        msg["To"] = to_email
        
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.login(self.username, self.password)
                server.send_message(msg)
            return True
        except Exception as e:
            print(f"發送郵件失敗: {str(e)}")
            return False
```

對應的整合測試（使用 unittest.mock）：

```python
# tests/services/test_email_service.py
import pytest
from unittest.mock import patch, MagicMock
from app.services.email_service import EmailService

@pytest.fixture
def email_service():
    return EmailService(
        smtp_server="test-smtp.example.com",
        smtp_port=587,
        username="test@example.com",
        password="testpassword"
    )

@patch("app.services.email_service.smtplib.SMTP")
def test_send_email_success(mock_smtp, email_service):
    # Arrange
    mock_server = MagicMock()
    mock_smtp.return_value.__enter__.return_value = mock_server
    
    to_email = "recipient@example.com"
    subject = "測試郵件"
    content = "這是一封測試郵件"
    
    # Act
    result = email_service.send_email(to_email, subject, content)
    
    # Assert
    assert result is True
    mock_server.login.assert_called_once_with(
        email_service.username, email_service.password
    )
    mock_server.send_message.assert_called_once()

@patch("app.services.email_service.smtplib.SMTP")
def test_send_email_failure(mock_smtp, email_service):
    # Arrange
    mock_smtp.return_value.__enter__.side_effect = Exception("連接失敗")
    
    # Act
    result = email_service.send_email("test@example.com", "測試", "內容")
    
    # Assert
    assert result is False
```

## 測試中間件

測試中間件是整合測試的重要部分。

### 測試策略

| 策略 | 說明 |
|------|------|
| **端到端流程測試** | 測試請求通過所有中間件的完整流程 |
| **依賴覆蓋** | 在測試中覆蓋特定依賴 |
| **上下文傳遞** | 測試上下文數據在中間件中的傳遞 |
| **錯誤處理** | 測試中間件的錯誤處理邏輯 |

### 示例：簡單錯誤處理中間件測試

假設我們有一個簡單的錯誤處理中間件：

```python
# app/middleware/error_handler.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            # 在實際應用中，這裡可能會記錄錯誤
            return JSONResponse(
                status_code=500,
                content={"detail": "發生內部服務器錯誤"}
            )
```

對應的整合測試：

```python
# tests/middleware/test_error_handler.py
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from app.middleware.error_handler import ErrorHandlerMiddleware

@pytest.fixture
def app_with_middleware():
    app = FastAPI()
    app.add_middleware(ErrorHandlerMiddleware)
    
    @app.get("/normal")
    def normal_route():
        return {"message": "正常路由"}
    
    @app.get("/http-error")
    def http_error_route():
        raise HTTPException(status_code=404, detail="找不到資源")
    
    @app.get("/exception")
    def exception_route():
        raise ValueError("測試異常")
    
    return app

@pytest.fixture
def client(app_with_middleware):
    return TestClient(app_with_middleware)

def test_normal_route(client):
    response = client.get("/normal")
    assert response.status_code == 200
    assert response.json() == {"message": "正常路由"}

def test_http_exception(client):
    # HTTPException 應該由 FastAPI 的異常處理器處理
    response = client.get("/http-error")
    assert response.status_code == 404
    assert response.json() == {"detail": "找不到資源"}

def test_unhandled_exception(client):
    # 未處理的異常應該由我們的中間件捕獲
    response = client.get("/exception")
    assert response.status_code == 500
    assert response.json() == {"detail": "發生內部服務器錯誤"}
```

## 整合測試的最佳實踐

### 測試環境隔離

| 實踐 | 說明 |
|------|------|
| **專用測試數據庫** | 使用獨立的測試數據庫，避免影響生產數據 |
| **測試後清理** | 每個測試後恢復環境到初始狀態 |
| **環境變量控制** | 使用環境變量區分測試和生產配置 |
| **容器化測試** | 使用 Docker 等工具提供隔離的測試環境 |

### 測試數據管理

| 實踐 | 說明 |
|------|------|
| **測試夾具** | 使用 pytest fixtures 創建和管理測試數據 |
| **工廠模式** | 使用工廠函數或類生成測試數據 |
| **數據填充腳本** | 創建可重用的數據填充腳本 |
| **參數化測試** | 使用不同的數據集測試相同的功能 |

### 依賴管理

| 實踐 | 說明 |
|------|------|
| **依賴注入** | 使用依賴注入使組件可測試 |
| **依賴覆蓋** | 在測試中覆蓋依賴以控制行為 |
| **模擬外部服務** | 模擬外部服務以避免實際調用 |
| **測試替身** | 使用 mock、stub 或 fake 對象替代真實依賴 |

## 總結

整合測試是確保 FastAPI 應用程序各組件正確協同工作的關鍵。通過測試 API 端點、中間件、數據庫交互和外部服務集成，你可以在早期發現組件之間的交互問題。

### 整合測試要點

| 方面 | 關鍵點 |
|------|--------|
| **測試範圍** | 專注於測試組件之間的交互<br>確保各層級之間的協同工作 |
| **測試環境** | 創建隔離的測試環境<br>使用測試替身替代不穩定的依賴 |
| **數據庫測試** | 使用專用測試數據庫<br>確保測試之間的數據隔離 |
| **API 測試** | 使用 TestClient 測試完整的請求/響應流程<br>測試各種輸入情況和錯誤處理 |
| **外部依賴** | 模擬外部服務以確保測試的可靠性<br>測試與外部系統的集成點 |

在下一章節中，我們將探討如何進行端到端測試，以確保整個應用程序從用戶界面到後端數據庫的正確運行。

---

通過合理的整合測試策略，你可以在保持測試速度和可維護性的同時，確保應用程序的各個組件能夠正確地協同工作。整合測試是單元測試和端到端測試之間的重要橋樑，為應用程序提供了全面的質量保障。