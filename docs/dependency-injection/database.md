# FastAPI 依賴注入實戰：資料庫管理

## 簡介

資料庫連接管理是 Web API 開發中的核心挑戰之一。如何有效地處理資料庫連接、事務和查詢，同時保持代碼的可測試性和可維護性，是每個開發者都需要面對的問題。FastAPI 的依賴注入系統為資料庫管理提供了優雅的解決方案，讓我們能夠以結構化和高效的方式處理資料庫操作。

本文將探討如何利用 FastAPI 的依賴注入機制實現各種資料庫管理模式，從基本的連接管理到複雜的事務處理和資源池化。

## 基本資料庫連接管理

### 1. 同步資料庫連接

最簡單的資料庫連接模式是為每個請求創建一個新的連接，並在請求結束時關閉它。

```python
from fastapi import FastAPI, Depends
import sqlite3
from contextlib import contextmanager

app = FastAPI()

@contextmanager
def get_db_connection():
    conn = sqlite3.connect("example.db")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def get_db():
    with get_db_connection() as conn:
        yield conn

@app.get("/users/{user_id}")
def read_user(user_id: int, db: sqlite3.Connection = Depends(get_db)):
    user = db.execute(
        "SELECT * FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    
    if user is None:
        return {"error": "User not found"}
    
    return dict(user)
```

**優點**：
- 簡單易懂
- 每個請求都有獨立的連接，避免並發問題

**缺點**：
- 頻繁創建和關閉連接可能影響性能
- 不適合高並發場景

### 2. 異步資料庫連接

對於支持異步的資料庫驅動，我們可以使用異步依賴實現更高效的連接管理。

```python
from fastapi import FastAPI, Depends
from databases import Database
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String

# 資料庫配置
DATABASE_URL = "sqlite:///./test.db"
database = Database(DATABASE_URL)

# SQLAlchemy 模型定義
metadata = MetaData()
users = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String),
    Column("email", String)
)

# 創建表（在實際應用中通常使用遷移工具）
engine = create_engine(DATABASE_URL)
metadata.create_all(engine)

app = FastAPI()

# 啟動和關閉事件
@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

# 依賴函數
async def get_db():
    return database

@app.get("/users/{user_id}")
async def read_user(user_id: int, db: Database = Depends(get_db)):
    query = users.select().where(users.c.id == user_id)
    user = await db.fetch_one(query)
    
    if user is None:
        return {"error": "User not found"}
    
    return dict(user)
```

**優點**：
- 支持異步操作，提高並發性能
- 連接在應用啟動時建立，避免頻繁創建連接

**缺點**：
- 需要使用支持異步的資料庫驅動
- 可能需要更複雜的錯誤處理

## 進階資料庫管理模式

### 1. 連接池管理

對於高並發應用，使用連接池可以顯著提高性能。

```python
from fastapi import FastAPI, Depends
import databases
import sqlalchemy

# 資料庫配置
DATABASE_URL = "postgresql://user:password@localhost/dbname"
database = databases.Database(DATABASE_URL)

# SQLAlchemy 模型
metadata = sqlalchemy.MetaData()
users = sqlalchemy.Table(
    "users",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("name", sqlalchemy.String),
    sqlalchemy.Column("email", sqlalchemy.String)
)

app = FastAPI()

@app.on_event("startup")
async def startup():
    # 連接池配置
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

async def get_db():
    return database

@app.get("/users/{user_id}")
async def read_user(user_id: int, db: databases.Database = Depends(get_db)):
    query = users.select().where(users.c.id == user_id)
    user = await db.fetch_one(query)
    
    if user is None:
        return {"error": "User not found"}
    
    return dict(user)
```

**優點**：
- 高效重用資料庫連接
- 適合高並發場景
- 避免連接泄漏

**缺點**：
- 需要謹慎配置池大小和超時
- 可能需要處理池耗盡的情況

### 2. 事務管理

對於需要原子性操作的場景，我們可以使用依賴注入實現事務管理。

```python
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel
from typing import List

# 資料庫配置
DATABASE_URL = "postgresql://user:password@localhost/dbname"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SQLAlchemy 模型
class UserModel(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)

# Pydantic 模型
class UserCreate(BaseModel):
    name: str
    email: str

class User(UserCreate):
    id: int
    
    class Config:
        orm_mode = True

app = FastAPI()

# 依賴函數
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 使用事務的路由
@app.post("/users/", response_model=User)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    # 檢查郵箱是否已存在
    db_user = db.query(UserModel).filter(UserModel.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # 創建新用戶
    db_user = UserModel(**user.dict())
    
    # 在事務中執行操作
    try:
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    return db_user
```

**優點**：
- 確保操作的原子性
- 自動處理提交和回滾
- 提供一致的資料視圖

**缺點**：
- 需要謹慎管理事務範圍
- 長時間事務可能影響並發性能

### 3. 資源庫模式 (Repository Pattern)

資源庫模式將資料庫操作封裝在專用類中，提供更好的關注點分離。

```python
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional

# 假設已定義 SQLAlchemy 模型和 get_db 依賴

# 用戶資源庫
class UserRepository:
    def __init__(self, db: Session):
        self.db = db
    
    def get_by_id(self, user_id: int) -> Optional[UserModel]:
        return self.db.query(UserModel).filter(UserModel.id == user_id).first()
    
    def get_by_email(self, email: str) -> Optional[UserModel]:
        return self.db.query(UserModel).filter(UserModel.email == email).first()
    
    def create(self, user: UserCreate) -> UserModel:
        db_user = UserModel(**user.dict())
        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)
        return db_user
    
    def list_all(self, skip: int = 0, limit: int = 100) -> List[UserModel]:
        return self.db.query(UserModel).offset(skip).limit(limit).all()

# 依賴函數
def get_user_repository(db: Session = Depends(get_db)) -> UserRepository:
    return UserRepository(db)

app = FastAPI()

@app.get("/users/{user_id}", response_model=User)
def read_user(
    user_id: int, 
    user_repo: UserRepository = Depends(get_user_repository)
):
    db_user = user_repo.get_by_id(user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@app.post("/users/", response_model=User)
def create_user(
    user: UserCreate, 
    user_repo: UserRepository = Depends(get_user_repository)
):
    db_user = user_repo.get_by_email(user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    return user_repo.create(user)
```

**優點**：
- 更好的關注點分離
- 提高代碼可讀性和可維護性
- 便於單元測試

**缺點**：
- 增加了額外的抽象層
- 可能導致代碼量增加

## 高級技巧與最佳實踐

### 1. 多資料庫支持

有時應用需要連接多個資料庫，我們可以使用依賴注入來管理多個資料庫連接。

```python
from fastapi import FastAPI, Depends
from databases import Database

app = FastAPI()

# 多資料庫配置
main_db = Database("postgresql://user:password@localhost/main_db")
analytics_db = Database("postgresql://user:password@localhost/analytics_db")

@app.on_event("startup")
async def startup():
    await main_db.connect()
    await analytics_db.connect()

@app.on_event("shutdown")
async def shutdown():
    await main_db.disconnect()
    await analytics_db.disconnect()

# 依賴函數
async def get_main_db():
    return main_db

async def get_analytics_db():
    return analytics_db

@app.get("/user-stats/{user_id}")
async def get_user_stats(
    user_id: int,
    main_db: Database = Depends(get_main_db),
    analytics_db: Database = Depends(get_analytics_db)
):
    # 從主資料庫獲取用戶資訊
    user = await main_db.fetch_one(
        "SELECT id, name FROM users WHERE id = :id", 
        values={"id": user_id}
    )
    
    if not user:
        return {"error": "User not found"}
    
    # 從分析資料庫獲取用戶統計數據
    stats = await analytics_db.fetch_one(
        "SELECT user_id, visit_count, last_visit FROM user_stats WHERE user_id = :id",
        values={"id": user_id}
    )
    
    # 組合結果
    return {
        "user": dict(user),
        "stats": dict(stats) if stats else {"visit_count": 0}
    }
```

**優點**：
- 支持多資料庫架構
- 清晰區分不同資料庫的用途
- 便於實現讀寫分離

**缺點**：
- 增加了系統複雜性
- 需要處理跨資料庫一致性問題

### 2. 動態資料庫選擇

在某些場景下，我們可能需要根據請求動態選擇資料庫連接。

```python
from fastapi import FastAPI, Depends, Header
from databases import Database
from typing import Dict, Optional

app = FastAPI()

# 資料庫連接池
db_pool: Dict[str, Database] = {
    "tenant1": Database("postgresql://user:password@localhost/tenant1"),
    "tenant2": Database("postgresql://user:password@localhost/tenant2"),
    "default": Database("postgresql://user:password@localhost/default")
}

@app.on_event("startup")
async def startup():
    for db in db_pool.values():
        await db.connect()

@app.on_event("shutdown")
async def shutdown():
    for db in db_pool.values():
        await db.disconnect()

# 依賴函數
async def get_tenant_db(x_tenant_id: Optional[str] = Header(None)):
    # 根據請求頭選擇租戶資料庫
    tenant_id = x_tenant_id or "default"
    return db_pool.get(tenant_id, db_pool["default"])

@app.get("/data")
async def read_data(db: Database = Depends(get_tenant_db)):
    # 從選定的租戶資料庫讀取數據
    data = await db.fetch_all("SELECT * FROM items LIMIT 10")
    return [dict(item) for item in data]
```

**優點**：
- 支持多租戶架構
- 根據請求上下文動態選擇資料庫
- 實現資料隔離

**缺點**：
- 需要管理多個資料庫連接
- 可能增加系統資源消耗

### 3. 資料庫遷移與版本管理

雖然不直接涉及依賴注入，但資料庫遷移是資料庫管理的重要部分。我們可以將遷移工具集成到 FastAPI 應用中。

```python
from fastapi import FastAPI, Depends
from alembic.config import Config
from alembic import command
import os

app = FastAPI()

# 資料庫遷移配置
def run_migrations():
    alembic_cfg = Config(os.path.join(os.path.dirname(__file__), "alembic.ini"))
    command.upgrade(alembic_cfg, "head")

# 應用啟動時運行遷移
@app.on_event("startup")
def startup_event():
    run_migrations()
    
# 其餘代碼與前面示例相似
```

**優點**：
- 自動化資料庫結構更新
- 確保應用和資料庫結構同步
- 支持版本回滾

**缺點**：
- 需要謹慎處理生產環境遷移
- 可能需要處理遷移衝突

## 測試資料庫依賴

測試資料庫依賴是確保應用穩定性的關鍵。以下是一些測試策略：

### 1. 使用測試資料庫

```python
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 測試資料庫配置
TEST_DATABASE_URL = "sqlite:///./test.db"
test_engine = create_engine(TEST_DATABASE_URL)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

# 創建測試資料庫和表
@pytest.fixture(scope="module")
def setup_test_db():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)

# 覆蓋依賴
@pytest.fixture
def client(setup_test_db):
    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as client:
        yield client
    
    app.dependency_overrides = {}

# 測試用例
def test_create_user(client):
    response = client.post(
        "/users/",
        json={"name": "Test User", "email": "test@example.com"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Test User"
    assert data["email"] == "test@example.com"
    assert "id" in data
```

### 2. 使用模擬對象

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_db():
    # 創建模擬資料庫會話
    mock_session = MagicMock()
    
    # 配置模擬查詢結果
    mock_user = MagicMock()
    mock_user.id = 1
    mock_user.name = "Test User"
    mock_user.email = "test@example.com"
    
    # 配置模擬查詢方法
    mock_query = MagicMock()
    mock_query.filter.return_value.first.return_value = mock_user
    mock_session.query.return_value = mock_query
    
    return mock_session

@pytest.fixture
def client(mock_db):
    def override_get_db():
        yield mock_db
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as client:
        yield client
    
    app.dependency_overrides = {}

def test_read_user(client, mock_db):
    response = client.get("/users/1")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == 1
    assert data["name"] == "Test User"
    assert data["email"] == "test@example.com"
    
    # 驗證模擬對象的調用
    mock_db.query.assert_called_once()
    mock_db.query().filter.assert_called_once()
```

## 最佳實踐

1. **使用連接池**：對於生產環境，總是使用連接池來管理資料庫連接。

2. **適當的錯誤處理**：確保資料庫錯誤被正確捕獲和處理，避免暴露敏感信息。

```python
@app.get("/users/{user_id}")
async def read_user(user_id: int, db: Database = Depends(get_db)):
    try:
        query = "SELECT * FROM users WHERE id = :id"
        user = await db.fetch_one(query=query, values={"id": user_id})
        
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        
        return dict(user)
    except Exception as e:
        # 記錄詳細錯誤，但向客戶端返回通用錯誤
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

3. **使用環境變數配置**：不要在代碼中硬編碼資料庫連接信息。

```python
import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./test.db")
    
    class Config:
        env_file = ".env"

settings = Settings()
database = Database(settings.database_url)
```

4. **定期關閉閒置連接**：避免資源泄漏。

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

async def close_idle_connections():
    # 假設 pool 是您的資料庫連接池
    await pool.close_idle_connections()

@app.on_event("startup")
async def start_scheduler():
    scheduler.add_job(close_idle_connections, "interval", minutes=10)
    scheduler.start()

@app.on_event("shutdown")
async def stop_scheduler():
    scheduler.shutdown()
```

5. **使用事務管理上下文**：確保資料庫操作的原子性。

```python
from fastapi import FastAPI, Depends, HTTPException
from databases import Database
from contextlib import asynccontextmanager

app = FastAPI()
database = Database("postgresql://user:password@localhost/dbname")

@asynccontextmanager
async def transaction(db: Database):
    transaction = await db.transaction()
    try:
        yield
    except Exception:
        await transaction.rollback()
        raise
    else:
        await transaction.commit()

@app.post("/transfer")
async def transfer_funds(
    from_id: int, 
    to_id: int, 
    amount: float, 
    db: Database = Depends(get_db)
):
    async with transaction(db):
        # 檢查餘額
        from_account = await db.fetch_one(
            "SELECT balance FROM accounts WHERE id = :id",
            values={"id": from_id}
        )
        
        if not from_account or from_account["balance"] < amount:
            raise HTTPException(status_code=400, detail="Insufficient funds")
        
        # 執行轉賬
        await db.execute(
            "UPDATE accounts SET balance = balance - :amount WHERE id = :id",
            values={"id": from_id, "amount": amount}
        )
        
        await db.execute(
            "UPDATE accounts SET balance = balance + :amount WHERE id = :id",
            values={"id": to_id, "amount": amount}
        )
        
        return {"message": "Transfer successful"}
```

## 結論

FastAPI 的依賴注入系統為資料庫管理提供了強大而靈活的解決方案。通過適當的依賴設計，我們可以實現高效的連接管理、事務處理和資源池化，同時保持代碼的可測試性和可維護性。

從基本的連接管理到高級的多租戶和分片架構，依賴注入都能幫助我們構建結構清晰、高效穩定的資料庫訪問層。通過遵循本文介紹的最佳實踐，您可以在 FastAPI 應用中實現既靈活又高效的資料庫管理。

記住，良好的資料庫管理不僅關乎性能，還關乎安全性和可維護性。合理使用依賴注入，可以幫助您在這些方面取得平衡，構建出更好的 FastAPI 應用。