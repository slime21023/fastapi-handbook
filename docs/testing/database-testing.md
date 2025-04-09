# 數據庫測試

## 數據庫測試的基本概念

數據庫測試是整合測試的重要組成部分，專注於驗證應用程序與數據庫之間的交互是否正確。在 FastAPI 應用中，數據庫測試確保 ORM 模型、存儲庫（repositories）和數據訪問層能夠正確執行 CRUD（創建、讀取、更新、刪除）操作。

| 概念 | 說明 |
|------|------|
| **測試範圍** | 數據庫模型、查詢和事務操作 |
| **隔離程度** | 使用專用測試數據庫或內存數據庫 |
| **執行速度** | 中等，受數據庫性能影響 |
| **測試數據** | 需要準備測試數據和清理機制 |
| **事務管理** | 使用事務確保測試之間的隔離 |

## 數據庫測試的挑戰與解決方案

| 挑戰 | 解決方案 |
|------|---------|
| **測試隔離** | 使用專用測試數據庫或內存數據庫 |
| **測試速度** | 使用內存數據庫加速測試執行 |
| **數據準備** | 使用夾具（fixtures）或工廠模式創建測試數據 |
| **數據清理** | 使用事務回滾或測試後清理策略 |
| **並行執行** | 確保測試可以並行運行而不互相干擾 |

## 設置測試數據庫環境

### 測試數據庫策略

在 FastAPI 應用中，有幾種常用的測試數據庫策略：

| 策略 | 優點 | 缺點 | 適用場景 |
|------|------|------|---------|
| **專用測試數據庫** | 與生產環境相似<br>可測試數據庫特定功能 | 設置複雜<br>測試較慢 | 完整的集成測試<br>測試數據庫特定功能 |
| **內存數據庫** | 速度快<br>無需外部依賴 | 可能缺少某些功能<br>與生產環境差異 | 單元測試<br>快速整合測試 |
| **測試容器** | 隔離性好<br>環境一致性 | 需要 Docker<br>資源消耗較大 | CI/CD 環境<br>完整的集成測試 |

### 使用 SQLite 內存數據庫進行測試

SQLite 內存數據庫是快速測試的理想選擇：

```python
# tests/conftest.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.main import app
from app.dependencies import get_db

# 創建內存數據庫引擎
TEST_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})

# 創建測試會話工廠
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")
def db():
    # 創建所有表
    Base.metadata.create_all(bind=engine)
    
    # 創建數據庫會話
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
    
    # 清理 - 刪除所有表
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
    
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c
    
    # 清理依賴覆蓋
    app.dependency_overrides = {}
```

## 測試數據準備

### 使用夾具創建測試數據

測試夾具（fixtures）是準備測試數據的理想方式：

```python
# tests/conftest.py
import pytest
from app.models import User, Item
from app.schemas import UserCreate, ItemCreate
from app.services.user_service import UserService
from app.services.item_service import ItemService

@pytest.fixture
def user_service(db):
    return UserService(db)

@pytest.fixture
def item_service(db):
    return ItemService(db)

@pytest.fixture
def test_user(user_service):
    user_data = UserCreate(
        username="testuser",
        email="test@example.com",
        password="password123"
    )
    user = user_service.create_user(user_data)
    return user

@pytest.fixture
def test_items(item_service, test_user):
    items = []
    # 創建測試商品
    for i in range(3):
        item_data = ItemCreate(
            name=f"測試商品 {i}",
            description=f"這是測試商品 {i} 的描述",
            price=10.0 * (i + 1),
            owner_id=test_user.id
        )
        items.append(item_service.create_item(item_data))
    return items
```

## 測試數據庫模型

### 測試模型關係和約束

```python
# tests/models/test_models.py
import pytest
from sqlalchemy.exc import IntegrityError
from app.models import User, Item

def test_user_model(db):
    # 創建用戶
    user = User(username="testuser", email="test@example.com", hashed_password="hashedpw")
    db.add(user)
    db.commit()
    
    # 驗證用戶被正確創建
    assert user.id is not None
    assert user.username == "testuser"
    assert user.email == "test@example.com"
    
    # 測試唯一性約束
    duplicate_user = User(username="testuser", email="test@example.com", hashed_password="hashedpw")
    db.add(duplicate_user)
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()

def test_item_model(db):
    # 創建用戶和商品
    user = User(username="itemowner", email="owner@example.com", hashed_password="hashedpw")
    db.add(user)
    db.commit()
    
    item = Item(name="測試商品", price=99.99, owner_id=user.id)
    db.add(item)
    db.commit()
    
    # 驗證商品被正確創建
    assert item.id is not None
    assert item.name == "測試商品"
    assert item.price == 99.99
    assert item.owner_id == user.id

def test_user_items_relationship(db):
    # 創建用戶
    user = User(username="relationuser", email="relation@example.com", hashed_password="hashedpw")
    db.add(user)
    db.commit()
    
    # 創建多個商品
    items = [
        Item(name="商品1", price=10.0, owner_id=user.id),
        Item(name="商品2", price=20.0, owner_id=user.id)
    ]
    db.add_all(items)
    db.commit()
    
    # 驗證關係
    db.refresh(user)
    assert len(user.items) == 2
    assert user.items[0].name == "商品1"
    assert user.items[1].name == "商品2"
    
    # 驗證反向關係
    assert items[0].owner.username == "relationuser"
```

## 測試存儲庫（Repository）

存儲庫模式是一種常見的數據訪問模式，它封裝了數據庫操作邏輯。

### 基本的存儲庫測試

```python
# tests/repositories/test_user_repository.py
import pytest
from app.repositories.user_repository import UserRepository
from app.models import User

@pytest.fixture
def user_repo(db):
    return UserRepository(db)

def test_create_user(user_repo):
    # 創建用戶
    user_data = {
        "username": "repouser",
        "email": "repo@example.com",
        "hashed_password": "hashedpassword"
    }
    user = user_repo.create(user_data)
    
    # 驗證用戶被創建
    assert user.id is not None
    assert user.username == user_data["username"]
    assert user.email == user_data["email"]

def test_get_user_by_id(user_repo):
    # 創建用戶
    user_data = {
        "username": "getuser",
        "email": "get@example.com",
        "hashed_password": "hashedpassword"
    }
    created_user = user_repo.create(user_data)
    
    # 獲取用戶
    user = user_repo.get_by_id(created_user.id)
    
    # 驗證
    assert user is not None
    assert user.id == created_user.id
    assert user.username == user_data["username"]

def test_update_user(user_repo):
    # 創建用戶
    user_data = {
        "username": "updateuser",
        "email": "update@example.com",
        "hashed_password": "hashedpassword"
    }
    user = user_repo.create(user_data)
    
    # 更新用戶
    update_data = {
        "username": "updateduser"
    }
    updated_user = user_repo.update(user.id, update_data)
    
    # 驗證
    assert updated_user.id == user.id
    assert updated_user.username == update_data["username"]
    assert updated_user.email == user_data["email"]  # 未更新的字段保持不變

def test_delete_user(user_repo):
    # 創建用戶
    user_data = {
        "username": "deleteuser",
        "email": "delete@example.com",
        "hashed_password": "hashedpassword"
    }
    user = user_repo.create(user_data)
    
    # 刪除用戶
    result = user_repo.delete(user.id)
    
    # 驗證
    assert result is True
    assert user_repo.get_by_id(user.id) is None
```

## 測試複雜查詢

對於複雜的數據庫查詢，需要準備更複雜的測試數據和更全面的驗證。

### 測試分頁和過濾

```python
# tests/repositories/test_item_repository.py
import pytest
from app.repositories.item_repository import ItemRepository
from app.repositories.user_repository import UserRepository

@pytest.fixture
def repos(db):
    return {
        "user": UserRepository(db),
        "item": ItemRepository(db)
    }

@pytest.fixture
def test_data(repos):
    # 創建測試用戶
    user1 = repos["user"].create({
        "username": "user1",
        "email": "user1@example.com",
        "hashed_password": "hashedpw"
    })
    
    # 創建測試商品
    items = []
    # 用戶1的商品
    for i in range(5):
        items.append(repos["item"].create({
            "name": f"商品{i}",
            "price": 10.0 * (i + 1),
            "category": "電子產品" if i % 2 == 0 else "家居",
            "owner_id": user1.id
        }))
    
    return {"user": user1, "items": items}

def test_get_items_with_pagination(repos, test_data):
    # 測試分頁
    items_page1 = repos["item"].get_all(skip=0, limit=3)
    items_page2 = repos["item"].get_all(skip=3, limit=3)
    
    assert len(items_page1) == 3
    assert len(items_page2) == 2  # 總共5個商品，第二頁只有2個
    
    # 驗證分頁數據不重複
    page1_ids = [item.id for item in items_page1]
    page2_ids = [item.id for item in items_page2]
    assert not any(item_id in page1_ids for item_id in page2_ids)

def test_get_items_by_category(repos, test_data):
    # 獲取電子產品類別的商品
    electronics = repos["item"].get_by_category("電子產品")
    assert len(electronics) == 3
    assert all(item.category == "電子產品" for item in electronics)
    
    # 獲取家居類別的商品
    home_items = repos["item"].get_by_category("家居")
    assert len(home_items) == 2
    assert all(item.category == "家居" for item in home_items)
```

## 測試事務和並發

測試數據庫事務和並發操作是確保數據一致性的重要部分。

### 測試事務操作

```python
# tests/repositories/test_transaction.py
import pytest
from app.repositories.order_repository import OrderRepository
from app.models import User, Item, Order

@pytest.fixture
def order_repo(db):
    return OrderRepository(db)

@pytest.fixture
def setup_order_data(db):
    # 創建用戶
    buyer = User(username="buyer", email="buyer@example.com", hashed_password="hashedpw")
    seller = User(username="seller", email="seller@example.com", hashed_password="hashedpw")
    db.add_all([buyer, seller])
    db.commit()
    
    # 創建商品
    item = Item(
        name="測試商品",
        price=100.0,
        inventory=5,
        owner_id=seller.id
    )
    db.add(item)
    db.commit()
    
    return {"buyer": buyer, "seller": seller, "item": item}

def test_create_order_success(order_repo, setup_order_data):
    buyer = setup_order_data["buyer"]
    item = setup_order_data["item"]
    
    # 執行訂單創建
    order = order_repo.create_order(
        buyer_id=buyer.id,
        item_id=item.id,
        quantity=2,
        amount=item.price * 2
    )
    
    # 驗證訂單記錄
    assert order.id is not None
    assert order.buyer_id == buyer.id
    assert order.item_id == item.id
    assert order.quantity == 2
    assert order.amount == 200.0
    assert order.status == "completed"
    
    # 驗證庫存更新
    assert item.inventory == 3  # 原庫存5，購買2個

def test_create_order_insufficient_inventory(order_repo, setup_order_data):
    buyer = setup_order_data["buyer"]
    item = setup_order_data["item"]
    
    # 嘗試購買超過庫存的數量
    with pytest.raises(ValueError) as exc_info:
        order_repo.create_order(
            buyer_id=buyer.id,
            item_id=item.id,
            quantity=10,  # 庫存只有5個
            amount=item.price * 10
        )
    
    assert "庫存不足" in str(exc_info.value)
    
    # 驗證庫存未變化
    assert item.inventory == 5
```

## 測試數據庫遷移

測試數據庫遷移確保模式變更不會破壞現有功能。

### 簡單的遷移測試

```python
# tests/migrations/test_migrations.py
import subprocess
import pytest
from sqlalchemy import inspect
from app.database import engine

def test_migrations_apply_successfully():
    # 運行遷移命令
    result = subprocess.run(
        ["alembic", "upgrade", "head"],
        capture_output=True,
        text=True
    )
    
    # 檢查命令是否成功執行
    assert result.returncode == 0, f"遷移失敗: {result.stderr}"

def test_table_structure():
    # 獲取檢查器
    inspector = inspect(engine)
    
    # 檢查表是否存在
    tables = inspector.get_table_names()
    assert "users" in tables
    assert "items" in tables
    
    # 檢查列
    user_columns = {col["name"] for col in inspector.get_columns("users")}
    assert "id" in user_columns
    assert "username" in user_columns
    assert "email" in user_columns
    assert "hashed_password" in user_columns
```

## 測試數據庫性能

對於性能關鍵的應用，測試數據庫查詢性能是很重要的。

### 簡單的性能測試

```python
# tests/performance/test_db_performance.py
import time
import pytest
from app.repositories.item_repository import ItemRepository
from app.models import User, Item

@pytest.fixture
def setup_performance_data(db):
    # 創建測試用戶
    user = User(username="perfuser", email="perf@example.com", hashed_password="hashedpw")
    db.add(user)
    db.commit()
    
    # 創建測試商品
    items = []
    for i in range(50):
        item = Item(
            name=f"性能測試商品 {i}",
            price=10.0 * (i % 10 + 1),
            category="電子產品" if i % 3 == 0 else "家居" if i % 3 == 1 else "辦公用品",
            owner_id=user.id
        )
        items.append(item)
    
    db.add_all(items)
    db.commit()
    
    return {"user": user, "items": items}

@pytest.fixture
def item_repo(db):
    return ItemRepository(db)

def test_get_all_items_performance(item_repo, setup_performance_data):
    # 測量獲取所有商品的性能
    start_time = time.time()
    items = item_repo.get_all(limit=100)
    end_time = time.time()
    
    duration = end_time - start_time
    assert len(items) >= 50
    assert duration < 0.1  # 期望查詢時間小於 100ms

def test_filter_by_category_performance(item_repo, setup_performance_data):
    # 測量按類別過濾的性能
    start_time = time.time()
    items = item_repo.get_by_category("電子產品")
    end_time = time.time()
    
    duration = end_time - start_time
    assert len(items) > 0
    assert duration < 0.05  # 期望查詢時間小於 50ms
```

## 數據庫測試的最佳實踐

### 測試隔離

| 實踐 | 說明 |
|------|------|
| **獨立測試數據庫** | 使用專用的測試數據庫，避免干擾生產數據 |
| **測試夾具作用域** | 合理設置夾具作用域，平衡性能和隔離性 |
| **事務回滾** | 使用事務確保測試之間的隔離 |
| **清理測試數據** | 測試後清理創建的數據，避免數據積累 |

### 測試數據管理

| 實踐 | 說明 |
|------|------|
| **工廠模式** | 使用工廠模式生成測試數據，提高可維護性 |
| **參數化測試** | 使用不同的數據集測試相同的功能 |
| **測試數據生成器** | 使用工具自動生成大量測試數據 |
| **真實數據子集** | 在某些情況下使用生產數據的子集進行測試 |

### 數據庫性能測試

| 實踐 | 說明 |
|------|------|
| **查詢性能基準** | 設置查詢性能的基準和閾值 |
| **索引測試** | 測試索引對查詢性能的影響 |
| **大數據集測試** | 使用大數據集測試系統在負載下的性能 |
| **慢查詢識別** | 識別並優化慢查詢 |

## 常見的數據庫測試模式

### 存儲庫模式測試

存儲庫模式是一種常見的數據訪問模式，它封裝了數據庫操作邏輯。測試存儲庫確保數據訪問層正確工作。

```python
# app/repositories/base_repository.py
class BaseRepository:
    def __init__(self, db):
        self.db = db
    
    def get_by_id(self, model, id):
        return self.db.query(model).filter(model.id == id).first()
    
    def get_all(self, model, skip=0, limit=100):
        return self.db.query(model).offset(skip).limit(limit).all()
    
    def create(self, model, data):
        instance = model(**data)
        self.db.add(instance)
        self.db.commit()
        self.db.refresh(instance)
        return instance
    
    def update(self, instance, data):
        for key, value in data.items():
            setattr(instance, key, value)
        self.db.commit()
        self.db.refresh(instance)
        return instance
    
    def delete(self, instance):
        self.db.delete(instance)
        self.db.commit()
        return True
```

測試這個基本存儲庫：

```python
# tests/repositories/test_base_repository.py
import pytest
from app.repositories.base_repository import BaseRepository
from app.models import User

@pytest.fixture
def base_repo(db):
    return BaseRepository(db)

def test_create_and_get_by_id(base_repo):
    # 創建用戶
    user_data = {
        "username": "baseuser",
        "email": "base@example.com",
        "hashed_password": "hashedpw"
    }
    user = base_repo.create(User, user_data)
    
    # 獲取用戶
    retrieved = base_repo.get_by_id(User, user.id)
    
    # 驗證
    assert retrieved.id == user.id
    assert retrieved.username == user_data["username"]
    assert retrieved.email == user_data["email"]

def test_get_all(base_repo):
    # 創建多個用戶
    users_data = [
        {"username": "user1", "email": "user1@example.com", "hashed_password": "pw1"},
        {"username": "user2", "email": "user2@example.com", "hashed_password": "pw2"},
        {"username": "user3", "email": "user3@example.com", "hashed_password": "pw3"}
    ]
    
    for data in users_data:
        base_repo.create(User, data)
    
    # 獲取所有用戶
    users = base_repo.get_all(User)
    
    # 驗證
    assert len(users) >= 3
    usernames = [user.username for user in users]
    assert "user1" in usernames
    assert "user2" in usernames
    assert "user3" in usernames

def test_update(base_repo):
    # 創建用戶
    user_data = {
        "username": "updateuser",
        "email": "update@example.com",
        "hashed_password": "hashedpw"
    }
    user = base_repo.create(User, user_data)
    
    # 更新用戶
    updated = base_repo.update(user, {"username": "updated"})
    
    # 驗證
    assert updated.id == user.id
    assert updated.username == "updated"
    assert updated.email == user_data["email"]  # 未更新的字段保持不變

def test_delete(base_repo):
    # 創建用戶
    user_data = {
        "username": "deleteuser",
        "email": "delete@example.com",
        "hashed_password": "hashedpw"
    }
    user = base_repo.create(User, user_data)
    
    # 刪除用戶
    result = base_repo.delete(user)
    
    # 驗證
    assert result is True
    assert base_repo.get_by_id(User, user.id) is None
```

## 總結

數據庫測試是確保應用程序數據層正確性和性能的關鍵。通過測試數據庫模型、關係、存儲庫和查詢性能，你可以確保應用程序的數據訪問層正確工作。

### 數據庫測試要點

| 方面 | 關鍵點 |
|------|--------|
| **測試環境** | 使用專用測試數據庫或內存數據庫<br>確保測試之間的隔離 |
| **測試數據** | 使用夾具或工廠模式創建測試數據<br>確保數據的一致性和可重複性 |
| **模型測試** | 測試模型關係和約束<br>確保模型行為符合預期 |
| **存儲庫測試** | 測試 CRUD 操作和複雜查詢<br>確保數據訪問層正確工作 |
| **性能測試** | 測試查詢性能<br>識別並優化慢查詢 |

通過全面的數據庫測試，你可以確保應用程序的數據層穩定可靠，為整個應用程序提供堅實的基礎。