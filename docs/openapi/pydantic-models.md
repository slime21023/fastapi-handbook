# Pydantic Models

## Pydantic 簡介

Pydantic 是一個數據驗證和設置管理庫，是 FastAPI 的核心組件之一。它使用 Python 類型註解來定義數據模型，提供了強大的數據驗證、轉換和文檔生成功能。Pydantic v2.0 帶來了顯著的性能提升和一些 API 變更。

| 核心功能 | 說明 |
|---------|------|
| **數據驗證** | 自動驗證數據符合定義的類型和約束 |
| **類型轉換** | 自動將輸入數據轉換為適當的類型 |
| **錯誤處理** | 提供清晰的驗證錯誤信息 |
| **IDE 支持** | 提供代碼補全和類型檢查支持 |

## 基本模型定義

### 創建基本模型

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class User(BaseModel):
    id: int
    name: str
    email: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    tags: List[str] = []
    description: Optional[str] = None
```

### 字段類型與默認值

| 特性 | 範例 | 說明 |
|------|------|------|
| **必填字段** | `name: str` | 沒有默認值的字段是必填的 |
| **可選字段** | `description: Optional[str] = None` | 有默認值的字段是可選的 |
| **默認值** | `is_active: bool = True` | 如果未提供值，則使用默認值 |
| **默認工廠** | `created_at: datetime = Field(default_factory=datetime.now)` | 使用函數生成默認值 |
| **複雜類型** | `tags: List[str] = []` | 支持嵌套的複雜類型 |

## 模型使用

### 創建模型實例

```python
# 從字典創建
user_data = {
    "id": 1,
    "name": "John Doe",
    "email": "john@example.com",
    "tags": ["admin", "user"]
}
user = User.model_validate(user_data)

# 直接創建
user = User(
    id=1,
    name="John Doe",
    email="john@example.com"
)

# 訪問字段
print(user.name)  # 輸出: John Doe
print(user.is_active)  # 輸出: True (默認值)
```

### 模型轉換

```python
# 轉換為字典
user_dict = user.model_dump()

# 轉換為 JSON 字符串
user_json = user.model_dump_json()

# 轉換為 JSON 字符串（帶格式化）
user_json_formatted = user.model_dump_json(indent=2)

# 排除某些字段
user_dict_partial = user.model_dump(exclude={"created_at", "tags"})

# 僅包含某些字段
user_dict_partial = user.model_dump(include={"id", "name", "email"})
```

### 模型複製與更新

```python
# 創建模型的副本並更新某些字段
user2 = user.model_copy(update={"name": "Jane Doe", "email": "jane@example.com"})

# 排除未設置的字段
user_dict = user.model_dump(exclude_unset=True)

# 排除默認值
user_dict = user.model_dump(exclude_defaults=True)

# 排除為 None 的值
user_dict = user.model_dump(exclude_none=True)
```

## 嵌套模型

### 定義嵌套模型

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class Image(BaseModel):
    url: str
    name: str
    width: Optional[int] = None
    height: Optional[int] = None

class Item(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    price: float
    images: List[Image] = []
```

### 使用嵌套模型

```python
item = Item(
    id=1,
    name="Smartphone",
    price=699.99,
    images=[
        Image(url="https://example.com/img1.jpg", name="Front view"),
        Image(url="https://example.com/img2.jpg", name="Back view", width=1000, height=800)
    ]
)

# 訪問嵌套字段
print(item.images[0].url)  # 輸出: https://example.com/img1.jpg
```

## 字段驗證

### 內建驗證器

Pydantic 提供了許多內建驗證器，可以通過 `Field` 函數應用：

```python
from pydantic import BaseModel, Field, EmailStr, HttpUrl

class User(BaseModel):
    id: int = Field(gt=0, description="用戶唯一標識符")
    name: str = Field(min_length=2, max_length=50)
    email: EmailStr
    website: HttpUrl = None
    age: int = Field(None, ge=18, lt=100)
    score: float = Field(0, ge=0, le=100)
```

| 驗證選項 | 適用類型 | 說明 |
|---------|---------|------|
| `min_length`/`max_length` | 字符串 | 字符串長度限制 |
| `pattern` | 字符串 | 正則表達式模式 |
| `gt`/`ge` | 數值 | 大於/大於等於 |
| `lt`/`le` | 數值 | 小於/小於等於 |
| `multiple_of` | 數值 | 必須是指定值的倍數 |

### 自定義驗證器

使用 `field_validator` 裝飾器創建自定義驗證邏輯：

```python
from pydantic import BaseModel, field_validator

class User(BaseModel):
    id: int
    name: str
    password: str
    password_confirm: str

    @field_validator('name')
    @classmethod
    def name_must_contain_space(cls, v):
        if ' ' not in v:
            raise ValueError('必須包含姓和名')
        return v.title()

    @field_validator('password_confirm')
    @classmethod
    def passwords_match(cls, v, info):
        if 'password' in info.data and v != info.data['password']:
            raise ValueError('密碼不匹配')
        return v
```

### 模型驗證器

使用 `model_validator` 進行跨字段驗證：

```python
from pydantic import BaseModel, model_validator

class Offer(BaseModel):
    original_price: float
    discount_price: float

    @model_validator(mode='after')
    def check_prices(self):
        if self.discount_price >= self.original_price:
            raise ValueError('折扣價必須低於原價')
        return self
```

## 配置與行為自定義

### 模型配置

使用 `model_config` 自定義模型行為：

```python
from pydantic import BaseModel, ConfigDict

class User(BaseModel):
    id: int
    name: str
    email: str
    password: str

    model_config = ConfigDict(
        # 允許從 ORM 對象創建模型
        from_attributes=True,
        
        # 允許額外字段（不在模型定義中的字段）
        extra='ignore',  # 或 'allow', 'forbid'
        
        # 大小寫敏感性
        case_sensitive=False,
        
        # 字段別名
        populate_by_name=True,
        
        # 在序列化時排除某些字段
        json_schema_extra={
            'json_schema_extra': {
                'examples': [
                    {
                        'id': 1,
                        'name': 'John Doe',
                        'email': 'john@example.com',
                        'password': '********'
                    }
                ]
            }
        }
    )
```

| 配置選項 | 說明 |
|---------|------|
| `from_attributes` | 允許從 ORM 對象創建模型（通過屬性訪問而非字典訪問） |
| `extra` | 控制額外字段的處理方式：`ignore`（忽略）、`allow`（允許）或 `forbid`（禁止） |
| `alias_generator` | 函數，用於自動生成字段別名 |
| `populate_by_name` | 允許通過字段名而非別名填充模型 |
| `validate_assignment` | 在賦值時驗證字段 |

### 字段別名

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    user_id: int = Field(alias='id')
    user_name: str = Field(alias='name')
```

## 高級類型

### 複雜類型註解

```python
from pydantic import BaseModel
from typing import Dict, List, Set, Tuple, Union, Optional

class AdvancedModel(BaseModel):
    # 字典類型
    metadata: Dict[str, str] = {}
    
    # 列表類型
    tags: List[str] = []
    
    # 集合類型（無重複元素）
    unique_ids: Set[int] = set()
    
    # 元組類型（固定長度和類型）
    coordinates: Tuple[float, float] = None
    
    # 聯合類型（多種可能類型之一）
    value: Union[int, str, bool] = None
    
    # 可選類型（等同於 Union[T, None]）
    description: Optional[str] = None
```

### 自定義類型與約束

```python
from pydantic import BaseModel, Field
from typing import Annotated

# 在 Pydantic v2 中，使用 Annotated 和 Field 替代 constr, conint, confloat
Username = Annotated[str, Field(min_length=3, max_length=50, pattern=r'^[a-zA-Z0-9_]+$')]
Age = Annotated[int, Field(ge=0, lt=120)]
Score = Annotated[float, Field(ge=0, le=100)]

class CustomTypesModel(BaseModel):
    # 受約束的字符串：長度限制，正則模式
    username: Username
    
    # 受約束的整數：範圍限制
    age: Age
    
    # 受約束的浮點數：範圍限制
    score: Score
```

## 實際應用範例

### API 請求和響應模型

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional

app = FastAPI()

# 請求模型
class UserCreate(BaseModel):
    name: str = Field(min_length=2)
    email: EmailStr
    password: str = Field(min_length=8)

# 響應模型（排除敏感字段）
class UserResponse(BaseModel):
    id: int
    name: str
    email: EmailStr
    is_active: bool

# 數據庫模型
class UserInDB(UserCreate):
    id: int
    hashed_password: str
    is_active: bool = True

# 模擬數據庫
fake_users_db = {}

@app.post("/users/", response_model=UserResponse)
async def create_user(user: UserCreate):
    # 檢查郵箱是否已存在
    if any(u.email == user.email for u in fake_users_db.values()):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # 模擬密碼哈希
    hashed_password = user.password + "_hashed"
    
    # 創建用戶記錄
    user_id = len(fake_users_db) + 1
    user_in_db = UserInDB(
        id=user_id,
        **user.model_dump(),
        hashed_password=hashed_password
    )
    
    # 保存到數據庫
    fake_users_db[user_id] = user_in_db
    
    # 返回用戶信息（不包含密碼）
    return user_in_db
```

### 數據轉換與處理

```python
from pydantic import BaseModel, field_validator
from datetime import datetime
from typing import List, Optional

class PostBase(BaseModel):
    title: str
    content: str
    published: bool = False

class PostCreate(PostBase):
    tags: List[str] = []

class PostUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    published: Optional[bool] = None
    tags: Optional[List[str]] = None

class PostInDB(PostBase):
    id: int
    author_id: int
    created_at: datetime
    updated_at: datetime
    tags: List[str] = []

    @field_validator('updated_at')
    @classmethod
    def set_updated_at(cls, v):
        return v or datetime.now()

class PostResponse(PostInDB):
    model_config = {
        "from_attributes": True
    }
```

## 與 ORM 集成

### SQLAlchemy 模型轉換

```python
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, ConfigDict
from datetime import datetime

# SQLAlchemy 模型
Base = declarative_base()

class UserDB(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)

# Pydantic 模型
class User(BaseModel):
    id: int
    name: str
    email: str
    is_active: bool
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

# 使用範例
def get_user(db_session, user_id: int):
    db_user = db_session.query(UserDB).filter(UserDB.id == user_id).first()
    if db_user:
        # 從 ORM 模型轉換為 Pydantic 模型
        return User.model_validate(db_user)
    return None
```

## 性能優化

### 使用 `model_construct` 跳過驗證

```python
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    email: str

# 當確定數據有效時，可以跳過驗證以提高性能
user = User.model_construct(id=1, name="John", email="john@example.com")
```

### 使用 RootModel 處理基本類型

```python
from pydantic import RootModel
from typing import List

# 創建一個整數列表的根模型
IntList = RootModel[List[int]]

# 使用根模型
numbers = IntList([1, 2, 3, 4, 5])
print(numbers.root)  # 輸出: [1, 2, 3, 4, 5]

# 添加一個數字
numbers.root.append(6)
print(numbers.root)  # 輸出: [1, 2, 3, 4, 5, 6]
```

## 總結

Pydantic v2.0 是 FastAPI 的核心組件，提供了強大的數據驗證和轉換功能，同時帶來了顯著的性能提升。通過使用 Pydantic 模型，您可以：

1. **定義數據結構**：使用 Python 類型註解清晰地定義數據模型
2. **驗證輸入數據**：自動驗證數據符合定義的類型和約束
3. **轉換數據格式**：在不同格式之間轉換數據（如 Python 對象、字典、JSON）
4. **生成文檔**：為 API 提供清晰的數據模型文檔

Pydantic v2.0 的主要 API 變更包括：
- `dict()` → `model_dump()`
- `json()` → `model_dump_json()`
- `copy()` → `model_copy()`
- `parse_obj()` → `model_validate()`
- `from_orm()` → `model_validate()` (with `from_attributes=True`)
- `validator()` → `field_validator()`
- `root_validator()` → `model_validator()`
- `Config` 類 → `model_config` 字典或 `ConfigDict`

這些變更使 Pydantic 在保持易用性的同時，提供了更好的性能和更一致的 API。