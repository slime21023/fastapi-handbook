# FastAPI 依賴注入的測試策略

## 簡介

測試是軟體開發中不可或缺的一部分，而 FastAPI 的依賴注入機制為我們提供了靈活且強大的測試能力。本文將介紹如何有效地測試使用依賴注入的 FastAPI 應用，包括單元測試、集成測試以及常見的測試模式。

## 依賴注入與測試的關係

依賴注入的一個主要優勢就是提高了代碼的可測試性。通過將依賴關係外部化，我們可以：

- 輕鬆替換真實依賴為測試替身（如 mock 或 stub）
- 隔離被測試的組件
- 模擬各種場景和錯誤情況

## 常用測試工具

在測試 FastAPI 應用時，以下工具非常有用：

- **pytest**：Python 的主流測試框架
- **TestClient**：FastAPI 提供的測試客戶端
- **unittest.mock**：Python 標準庫中的 mock 工具
- **dependency_overrides**：FastAPI 的依賴覆蓋機制

## 測試策略

### 1. 單元測試依賴函數

首先，我們應該獨立測試依賴函數，確保它們的邏輯正確。

```python
# 被測試的依賴函數
def get_db():
    db = Database()
    try:
        yield db
    finally:
        db.close()

# 測試
def test_get_db():
    db_generator = get_db()
    db = next(db_generator)
    assert isinstance(db, Database)
    
    # 測試清理邏輯
    try:
        next(db_generator)
    except StopIteration:
        pass  # 預期的行為
```

### 2. 使用依賴覆蓋進行路由測試

FastAPI 提供了 `app.dependency_overrides` 字典，允許我們在測試中替換依賴。

```python
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

# 應用代碼
app = FastAPI()

def get_db():
    # 實際應用中的資料庫連接邏輯
    ...

@app.get("/users/{user_id}")
def read_user(user_id: int, db = Depends(get_db)):
    return db.get_user(user_id)

# 測試代碼
def test_read_user():
    # 創建 mock 資料庫
    mock_db = MagicMock()
    mock_db.get_user.return_value = {"id": 1, "name": "Test User"}
    
    # 覆蓋依賴
    app.dependency_overrides[get_db] = lambda: mock_db
    
    client = TestClient(app)
    response = client.get("/users/1")
    
    assert response.status_code == 200
    assert response.json() == {"id": 1, "name": "Test User"}
    mock_db.get_user.assert_called_once_with(1)
    
    # 清理
    app.dependency_overrides = {}
```

### 3. 測試鏈式依賴

對於鏈式依賴，我們可以選擇覆蓋整個鏈或僅覆蓋鏈中的特定部分。

```python
# 應用代碼
def get_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401)
    return authorization

def get_current_user(token: str = Depends(get_token)):
    # 驗證 token 並返回用戶
    ...

@app.get("/me")
def read_me(user: dict = Depends(get_current_user)):
    return user

# 測試代碼 - 覆蓋整個鏈
def test_read_me():
    mock_user = {"id": 1, "name": "Test User"}
    app.dependency_overrides[get_current_user] = lambda: mock_user
    
    client = TestClient(app)
    response = client.get("/me")
    
    assert response.status_code == 200
    assert response.json() == mock_user
```

### 4. 使用 pytest fixtures 管理依賴覆蓋

使用 pytest fixtures 可以更好地組織測試代碼和依賴覆蓋。

```python
import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def client_with_mock_db():
    mock_db = MagicMock()
    mock_db.get_user.return_value = {"id": 1, "name": "Test User"}
    
    app.dependency_overrides[get_db] = lambda: mock_db
    
    client = TestClient(app)
    yield client, mock_db
    
    # 清理
    app.dependency_overrides = {}

def test_read_user_with_fixture(client_with_mock_db):
    client, mock_db = client_with_mock_db
    
    response = client.get("/users/1")
    
    assert response.status_code == 200
    assert response.json() == {"id": 1, "name": "Test User"}
    mock_db.get_user.assert_called_once_with(1)
```

### 5. 測試類別依賴

對於類別依賴，我們可以創建測試替身或覆蓋特定方法。

```python
# 應用代碼
class UserService:
    def get_user(self, user_id: int):
        # 實際實現
        ...

@app.get("/users/{user_id}")
def read_user(user_id: int, service: UserService = Depends(UserService)):
    return service.get_user(user_id)

# 測試代碼
def test_read_user_with_service():
    # 創建測試替身
    class TestUserService:
        def get_user(self, user_id: int):
            return {"id": user_id, "name": "Test User"}
    
    app.dependency_overrides[UserService] = TestUserService
    
    client = TestClient(app)
    response = client.get("/users/1")
    
    assert response.status_code == 200
    assert response.json() == {"id": 1, "name": "Test User"}
    
    app.dependency_overrides = {}
```

## 進階測試技巧

### 1. 模擬異步依賴

對於異步依賴，我們需要使用異步 mock 或創建異步測試替身。

```python
# 應用代碼
async def get_async_db():
    db = AsyncDatabase()
    try:
        yield db
    finally:
        await db.close()

# 測試代碼
async def mock_async_db():
    mock_db = AsyncMock()
    mock_db.get_user.return_value = {"id": 1, "name": "Test User"}
    yield mock_db

def test_async_dependency():
    app.dependency_overrides[get_async_db] = mock_async_db
    
    client = TestClient(app)
    response = client.get("/users/1")
    
    assert response.status_code == 200
    # 更多斷言...
```

### 2. 測試帶有子依賴的路由

對於使用 `dependencies` 參數的路由，我們同樣可以覆蓋這些依賴。

```python
# 應用代碼
def verify_admin(token: str = Depends(get_token)):
    # 驗證是否為管理員
    ...

@app.get("/admin", dependencies=[Depends(verify_admin)])
def admin_route():
    return {"message": "Admin access"}

# 測試代碼
def test_admin_route():
    # 覆蓋驗證依賴
    app.dependency_overrides[verify_admin] = lambda: True
    
    client = TestClient(app)
    response = client.get("/admin")
    
    assert response.status_code == 200
    assert response.json() == {"message": "Admin access"}
```

### 3. 集成測試與真實依賴

有時我們需要進行集成測試，使用真實的依賴（如測試資料庫）。

```python
@pytest.fixture
def test_db():
    # 設置測試資料庫
    db = Database("test_db")
    db.create_tables()
    
    yield db
    
    # 清理
    db.drop_tables()
    db.close()

def test_integration_with_db(test_db):
    # 使用真實的測試資料庫進行測試
    app.dependency_overrides[get_db] = lambda: test_db
    
    client = TestClient(app)
    # 執行測試...
    
    app.dependency_overrides = {}
```

## 最佳實踐

1. **隔離測試**：每個測試應該獨立運行，不依賴其他測試的狀態。

2. **清理覆蓋**：測試完成後恢復 `dependency_overrides`，避免影響其他測試。

3. **使用 fixtures**：利用 pytest fixtures 管理測試資源和依賴覆蓋。

4. **測試邊界情況**：不僅測試正常流程，也要測試錯誤處理和邊界情況。

5. **保持測試簡單**：每個測試應該專注於一個功能點，避免過於複雜的測試。

## 結論

FastAPI 的依賴注入系統為測試提供了強大的支持。通過依賴覆蓋機制，我們可以輕鬆隔離被測試的組件，模擬各種場景，確保我們的應用在各種情況下都能正常工作。良好的測試策略不僅能提高代碼質量，還能增強我們對代碼的信心，使重構和新功能開發更加安全。
