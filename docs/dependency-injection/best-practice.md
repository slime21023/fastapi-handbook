# FastAPI 依賴注入最佳實踐

## 簡介

依賴注入是 FastAPI 中最強大的功能之一，它能讓你的程式碼更加模組化、可測試且易於維護。然而，就像任何強大的工具一樣，使用不當可能會導致複雜性增加和維護困難。本文將分享一系列實用的最佳實踐，幫助你充分利用 FastAPI 的依賴注入系統，同時避免常見的陷阱。

## 依賴注入的基本原則

在深入探討具體實踐之前，讓我們先回顧依賴注入的核心原則：

1. **關注點分離**：每個依賴應該專注於單一職責
2. **可測試性**：依賴設計應便於在測試中替換或模擬
3. **可重用性**：依賴應該能夠在多個路由或其他依賴中重複使用
4. **明確性**：依賴關係應該清晰可見，而非隱藏在實現細節中

遵循這些原則將幫助你建立一個健壯且易於維護的應用架構。

## 依賴設計最佳實踐

### 1. 保持依賴函數簡潔明確

每個依賴函數應該有一個明確的目的，並且只做一件事。

**不推薦**:
```python
def get_user_and_validate_permissions(user_id: int, db: Session = Depends(get_db)):
    # 獲取用戶
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # 檢查權限
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Inactive user")
    if not user.has_permission("read:items"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    return user
```

**推薦**:
```python
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

def validate_active_user(user: User = Depends(get_user)):
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Inactive user")
    return user

def validate_user_permission(
    permission: str,
    user: User = Depends(validate_active_user)
):
    if not user.has_permission(permission):
        raise HTTPException(
            status_code=403,
            detail=f"User lacks required permission: {permission}"
        )
    return user

# 使用方式
@app.get("/items/{item_id}")
def read_item(
    item_id: int,
    user: User = Depends(lambda: validate_user_permission("read:items"))
):
    # 處理請求
    return {"item_id": item_id, "owner": user.username}
```

這種方法讓每個依賴都專注於單一職責，並且可以靈活組合使用。

### 2. 使用依賴層次結構

構建依賴的層次結構，而不是將所有邏輯放在單一依賴中。

```python
# 基礎層：資料庫連接
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 資源層：資料訪問
def get_user_repository(db: Session = Depends(get_db)):
    return UserRepository(db)

def get_item_repository(db: Session = Depends(get_db)):
    return ItemRepository(db)

# 服務層：業務邏輯
def get_auth_service(
    user_repo: UserRepository = Depends(get_user_repository)
):
    return AuthService(user_repo)

def get_item_service(
    item_repo: ItemRepository = Depends(get_item_repository),
    user_repo: UserRepository = Depends(get_user_repository)
):
    return ItemService(item_repo, user_repo)

# 控制層：路由處理
@app.get("/items/{item_id}")
def read_item(
    item_id: int,
    item_service: ItemService = Depends(get_item_service)
):
    return item_service.get_item(item_id)
```

這種分層方法使得代碼結構清晰，每一層都有明確的職責。

### 3. 使用類作為依賴

對於複雜的依賴，使用類而不是函數可以提供更好的組織結構和可維護性。

```python
class UserPermission:
    def __init__(self, db: Session = Depends(get_db)):
        self.db = db
    
    def __call__(self, user_id: int):
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if not user.is_active:
            raise HTTPException(status_code=403, detail="Inactive user")
        return user
    
    def has_permission(self, permission: str):
        def check_permission(user: User = Depends(self)):
            if not user.has_permission(permission):
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission denied: {permission}"
                )
            return user
        return check_permission

# 使用方式
user_permission = UserPermission()

@app.get("/users/me")
def read_user(user: User = Depends(user_permission)):
    return user

@app.get("/admin/dashboard")
def admin_dashboard(
    user: User = Depends(user_permission.has_permission("admin"))
):
    return {"message": "Welcome to admin dashboard"}
```

類依賴允許你封裝相關功能，並提供更靈活的配置選項。

### 4. 使用 Pydantic 模型進行配置管理

使用 Pydantic 模型來管理應用配置，並將其作為依賴提供。

```python
from pydantic import BaseSettings, Field
from functools import lru_cache

class Settings(BaseSettings):
    database_url: str = Field(..., env="DATABASE_URL")
    api_key: str = Field(..., env="API_KEY")
    debug: bool = Field(False, env="DEBUG")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings():
    return Settings()

@app.get("/info")
def get_info(settings: Settings = Depends(get_settings)):
    if settings.debug:
        return {
            "database_url": settings.database_url,
            "api_key": settings.api_key,
            "debug": settings.debug
        }
    return {"status": "production"}
```

這種方法提供了類型安全的配置管理，並支持從環境變數或 .env 文件加載配置。

## 性能優化最佳實踐

### 1. 緩存重複使用的依賴

對於計算成本高但結果不常變化的依賴，使用緩存可以提高性能。

```python
from functools import lru_cache

@lru_cache()
def get_settings():
    # 這個函數只會被調用一次，結果會被緩存
    print("Loading settings...")  # 這行只會打印一次
    return Settings()

# 對於需要請求特定信息的依賴，可以使用參數化緩存
class UserService:
    @lru_cache(maxsize=100)
    def get_user_preferences(self, user_id: int):
        # 這個方法會為每個不同的 user_id 緩存一次結果
        print(f"Loading preferences for user {user_id}")
        # 假設這是一個昂貴的資料庫查詢
        return {"theme": "dark", "language": "zh-TW"}

def get_user_service():
    return UserService()

@app.get("/users/{user_id}/preferences")
def read_preferences(
    user_id: int,
    user_service: UserService = Depends(get_user_service)
):
    return user_service.get_user_preferences(user_id)
```

注意：使用 `lru_cache` 時要小心，確保緩存的數據不會過時或消耗過多內存。

### 2. 避免不必要的依賴嵌套

過度嵌套的依賴可能導致性能問題和難以跟踪的錯誤。

**不推薦**:
```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_user_repo(db: Session = Depends(get_db)):
    return UserRepository(db)

def get_auth_service(user_repo = Depends(get_user_repo)):
    return AuthService(user_repo)

def get_current_user(
    token: str = Depends(oauth2_scheme),
    auth_service = Depends(get_auth_service)
):
    return auth_service.get_current_user(token)

def get_current_active_user(
    current_user = Depends(get_current_user)
):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def get_current_admin_user(
    current_user = Depends(get_current_active_user)
):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not an admin")
    return current_user

# 路由使用了多層嵌套的依賴
@app.get("/admin/users")
def get_all_users(
    admin: User = Depends(get_current_admin_user),
    user_repo = Depends(get_user_repo)  # 重複依賴
):
    return user_repo.list_users()
```

**推薦**:
```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 將相關的依賴合併到一個服務類中
class UserService:
    def __init__(
        self, 
        db: Session = Depends(get_db),
        token: str = Depends(oauth2_scheme)
    ):
        self.db = db
        self.repo = UserRepository(db)
        self.token = token
    
    def get_current_user(self):
        # 驗證令牌並返回當前用戶
        user = self.repo.get_user_by_token(self.token)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials"
            )
        return user
    
    def require_active_user(self):
        user = self.get_current_user()
        if not user.is_active:
            raise HTTPException(status_code=400, detail="Inactive user")
        return user
    
    def require_admin(self):
        user = self.require_active_user()
        if not user.is_admin:
            raise HTTPException(status_code=403, detail="Not an admin")
        return user
    
    def list_users(self):
        return self.repo.list_users()

# 路由更簡潔，依賴更少
@app.get("/admin/users")
def get_all_users(service: UserService = Depends()):
    admin = service.require_admin()
    return service.list_users()
```

這種方法減少了依賴層數，並避免了重複解析相同的依賴。

## 測試最佳實踐

### 1. 設計便於測試的依賴

依賴應該易於在測試中替換或模擬。

```python
# 定義接口
from abc import ABC, abstractmethod

class EmailSender(ABC):
    @abstractmethod
    async def send_email(self, to: str, subject: str, body: str):
        pass

# 實現
class SmtpEmailSender(EmailSender):
    async def send_email(self, to: str, subject: str, body: str):
        # 實際發送郵件的邏輯
        pass

# 依賴提供者
def get_email_sender() -> EmailSender:
    return SmtpEmailSender()

# 使用依賴的服務
class NotificationService:
    def __init__(self, email_sender: EmailSender = Depends(get_email_sender)):
        self.email_sender = email_sender
    
    async def notify_user(self, user_email: str, message: str):
        await self.email_sender.send_email(
            to=user_email,
            subject="Notification",
            body=message
        )

# 在測試中使用模擬實現
class MockEmailSender(EmailSender):
    def __init__(self):
        self.sent_emails = []
    
    async def send_email(self, to: str, subject: str, body: str):
        self.sent_emails.append({"to": to, "subject": subject, "body": body})
```

### 2. 使用依賴覆蓋進行測試

FastAPI 提供了依賴覆蓋機制，使得測試變得簡單。

```python
from fastapi.testclient import TestClient
import pytest

@pytest.fixture
def mock_email_sender():
    return MockEmailSender()

@pytest.fixture
def client(mock_email_sender):
    app.dependency_overrides[get_email_sender] = lambda: mock_email_sender
    
    with TestClient(app) as client:
        yield client
    
    # 清理覆蓋
    app.dependency_overrides = {}

def test_send_notification(client, mock_email_sender):
    response = client.post(
        "/notify",
        json={"email": "user@example.com", "message": "Test notification"}
    )
    
    assert response.status_code == 200
    assert len(mock_email_sender.sent_emails) == 1
    assert mock_email_sender.sent_emails[0]["to"] == "user@example.com"
    assert "Test notification" in mock_email_sender.sent_emails[0]["body"]
```

### 3. 使用工廠模式簡化測試配置

工廠模式可以幫助你在測試中更容易地創建和配置依賴。

```python
# 依賴工廠
class DatabaseFactory:
    @staticmethod
    def get_test_db():
        # 返回測試資料庫會話
        engine = create_engine("sqlite:///./test.db")
        TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    @staticmethod
    def get_production_db():
        # 返回生產資料庫會話
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

# 根據環境選擇適當的工廠方法
def get_db():
    if settings.testing:
        return DatabaseFactory.get_test_db()
    return DatabaseFactory.get_production_db()

# 在測試中
@pytest.fixture
def override_get_db():
    return DatabaseFactory.get_test_db

@pytest.fixture
def client(override_get_db):
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as client:
        yield client
    
    app.dependency_overrides = {}
```

## 安全性最佳實踐

### 1. 分層身份驗證依賴

構建分層的身份驗證依賴，以實現更精細的訪問控制。

```python
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# 基本身份驗證
async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = get_user_by_username(username)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    
    return user

# 活躍用戶驗證
async def get_active_user(current_user = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# 基於角色的訪問控制
def RoleChecker(allowed_roles: List[str]):
    async def check_role(user = Depends(get_active_user)):
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=403, 
                detail=f"Role {user.role} not permitted"
            )
        return user
    return check_role

# 使用方式
@app.get("/users/me")
async def read_users_me(current_user = Depends(get_active_user)):
    return current_user

@app.get("/admin/dashboard")
async def admin_dashboard(
    current_user = Depends(RoleChecker(["admin", "superuser"]))
):
    return {"message": "Welcome to admin dashboard"}
```

### 2. 安全地處理敏感配置

確保敏感配置不會被意外暴露。

```python
from pydantic import BaseSettings, SecretStr

class Settings(BaseSettings):
    database_url: str
    secret_key: SecretStr  # 使用 SecretStr 保護敏感值
    
    class Config:
        env_file = ".env"

settings = Settings()

def get_db_url():
    # 安全地訪問資料庫 URL
    return settings.database_url

def get_jwt_settings():
    # 返回 JWT 設置，但不直接暴露密鑰
    return {
        "algorithm": "HS256",
        "access_token_expire_minutes": 30
    }

# 在需要實際密鑰的地方
def create_access_token(data: dict):
    # 只在需要時獲取實際密鑰值
    secret_key = settings.secret_key.get_secret_value()
    # 使用密鑰創建令牌
    # ...
```

## 組織和結構最佳實踐

### 1. 按功能組織依賴

將相關的依賴組織在一起，以提高可維護性。

```
project/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── dependencies/
│   │   ├── __init__.py
│   │   ├── database.py      # 資料庫依賴
│   │   ├── auth.py          # 身份驗證依賴
│   │   ├── services.py      # 外部服務依賴
│   │   └── commons.py       # 通用依賴
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── users.py
│   │   └── items.py
│   └── ...
```

在 `dependencies/__init__.py` 中導出常用依賴:

```python
# dependencies/__init__.py
from .database import get_db
from .auth import get_current_user, get_active_user
from .services import get_email_service, get_payment_service

__all__ = [
    "get_db", 
    "get_current_user", 
    "get_active_user",
    "get_email_service",
    "get_payment_service"
]
```

然後在路由中導入:

```python
# routers/users.py
from fastapi import APIRouter, Depends
from ..dependencies import get_db, get_current_user

router = APIRouter()

@router.get("/users/me")
def read_users_me(current_user = Depends(get_current_user)):
    return current_user
```

### 2. 使用依賴容器

對於大型應用，考慮使用依賴容器來管理複雜的依賴關係。

```python
from fastapi import Depends

class Container:
    def __init__(self):
        self._services = {}
    
    def register(self, name, factory):
        self._services[name] = factory
    
    def resolve(self, name):
        if name not in self._services:
            raise ValueError(f"Service {name} not registered")
        
        factory = self._services[name]
        return factory()

# 創建全局容器
container = Container()

# 註冊服務
container.register("db", get_db)
container.register("user_repository", lambda: UserRepository(next(container.resolve("db"))))
container.register("auth_service", lambda: AuthService(container.resolve("user_repository")))

# 依賴函數
def get_auth_service():
    return container.resolve("auth_service")

@app.get("/users/me")
def read_users_me(auth_service = Depends(get_auth_service)):
    current_user = auth_service.get_current_user()
    return current_user
```

這種方法在大型應用中特別有用，可以集中管理依賴關係。

## 常見陷阱與如何避免

### 1. 循環依賴

循環依賴是一個常見問題，可能導致難以診斷的錯誤。

**問題**:
```python
def get_service_a(service_b = Depends(get_service_b)):
    return ServiceA(service_b)

def get_service_b(service_a = Depends(get_service_a)):  # 循環依賴!
    return ServiceB(service_a)
```

**解決方案**:
1. 重構依賴關係，消除循環
2. 使用延遲初始化或工廠模式

```python
# 方案 1: 重構依賴關係
def get_common_dependency():
    return CommonDependency()

def get_service_a(common = Depends(get_common_dependency)):
    return ServiceA(common)

def get_service_b(common = Depends(get_common_dependency)):
    return ServiceB(common)

# 方案 2: 使用工廠模式
class ServiceFactory:
    _service_a = None
    _service_b = None
    
    @classmethod
    def get_service_a(cls):
        if cls._service_a is None:
            cls._service_a = ServiceA(cls.get_service_b)
        return cls._service_a
    
    @classmethod
    def get_service_b(cls):
        if cls._service_b is None:
            cls._service_b = ServiceB()
        return cls._service_b

def get_service_a():
    return ServiceFactory.get_service_a()

def get_service_b():
    return ServiceFactory.get_service_b()
```

### 2. 過度使用全局狀態

過度依賴全局狀態會使應用難以測試和維護。

**問題**:
```python
# 全局變數被多個依賴修改
db_connection = None

def get_db():
    global db_connection
    if db_connection is None:
        db_connection = create_connection()
    return db_connection

def close_db():
    global db_connection
    if db_connection:
        db_connection.close()
        db_connection = None
```

**解決方案**:
使用適當的生命週期管理和依賴注入，而不是全局變數。

```python
# 使用應用狀態和事件處理器
app = FastAPI()

@app.on_event("startup")
async def startup():
    app.state.db_pool = await create_db_pool()

@app.on_event("shutdown")
async def shutdown():
    await app.state.db_pool.close()

async def get_db():
    async with app.state.db_pool.acquire() as connection:
        yield connection
```

### 3. 忽略資源清理

未能正確清理資源可能導致資源洩漏。

**問題**:
```python
def get_db():
    db = SessionLocal()
    return db  # 沒有關閉連接!
```

**解決方案**:
使用上下文管理器或 `yield` 依賴確保資源被適當清理。

```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

## 依賴注入進階技巧

### 1. 條件依賴

根據條件選擇不同的依賴實現。

```python
def get_cache_strategy(request: Request):
    if "mobile" in request.headers.get("user-agent", "").lower():
        # 移動設備使用較小的緩存
        return LimitedCache(max_size=100)
    # 桌面設備使用較大的緩存
    return StandardCache(max_size=1000)

@app.get("/data")
def get_cached_data(cache = Depends(get_cache_strategy)):
    # 使用選定的緩存策略
    if result := cache.get("data"):
        return result
    
    # 生成新數據
    result = generate_data()
    cache.set("data", result)
    return result
```

### 2. 參數化依賴

創建可接受參數的依賴，以增加靈活性。

```python
def Paginator(
    default_page_size: int = 10,
    max_page_size: int = 100
):
    def paginate(
        page: int = 1,
        page_size: int = default_page_size
    ):
        if page < 1:
            page = 1
        
        if page_size > max_page_size:
            page_size = max_page_size
        
        return {
            "page": page,
            "page_size": page_size,
            "offset": (page - 1) * page_size,
            "limit": page_size
        }
    
    return paginate

# 使用方式
@app.get("/users")
def list_users(
    pagination = Depends(Paginator(default_page_size=20)),
    db: Session = Depends(get_db)
):
    users = db.query(User).offset(pagination["offset"]).limit(pagination["limit"]).all()
    return {
        "page": pagination["page"],
        "page_size": pagination["page_size"],
        "total": db.query(User).count(),
        "users": users
    }
```

### 3. 動態依賴解析

在運行時動態選擇依賴。

```python
class DynamicDependency:
    def __init__(self):
        self.implementations = {}
    
    def register(self, name, dependency):
        self.implementations[name] = dependency
    
    def get(self, name):
        if name not in self.implementations:
            raise ValueError(f"Implementation {name} not found")
        
        return self.implementations[name]

# 創建動態依賴解析器
dynamic_deps = DynamicDependency()

# 註冊不同的實現
dynamic_deps.register("memory_cache", MemoryCache())
dynamic_deps.register("redis_cache", RedisCache())

def get_cache(cache_type: str = "memory_cache"):
    return dynamic_deps.get(cache_type)

@app.get("/data")
def get_data(cache = Depends(get_cache)):
    # 使用選定的緩存實現
    return cache.get_data()

# 可以通過查詢參數選擇實現
# GET /data?cache_type=redis_cache
```

## 結論

FastAPI 的依賴注入系統是其最強大的功能之一，它能夠幫助你構建模組化、可測試且易於維護的應用。通過遵循本文中的最佳實踐，你可以充分利用這個系統的優勢，同時避免常見的陷阱。

關鍵要點回顧：

1. **保持依賴簡單明確**：每個依賴應該有一個明確的目的
2. **構建依賴層次結構**：使用分層方法組織依賴
3. **優化性能**：適當使用緩存和避免不必要的依賴嵌套
4. **設計便於測試的依賴**：使依賴易於在測試中替換或模擬
5. **安全處理敏感配置**：確保敏感信息不被意外暴露
6. **組織良好的代碼結構**：按功能組織依賴，提高可維護性

依賴注入是一種強大的模式，掌握它需要時間和實踐。隨著你的應用變得