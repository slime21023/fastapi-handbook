# OpenAPI Schema 進階用法

## 1. 複雜數據模型與 Schema 定制

### 1.1 複雜嵌套模型

FastAPI 支持複雜的嵌套模型，自動轉換為 OpenAPI Schema：

```python
from typing import List, Optional, Set
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl

class Image(BaseModel):
    url: HttpUrl
    name: str

class Item(BaseModel):
    name: str
    price: float
    images: Optional[List[Image]] = None
    tags: Set[str] = set()

app = FastAPI()

@app.post("/items/")
async def create_item(item: Item):
    return item
```

### 1.2 自定義 JSON Schema 屬性

使用 Pydantic 的 `Field` 和 `model_config` 自定義 Schema：

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field

class Item(BaseModel):
    name: str = Field(..., title="名稱", example="Foo")
    price: float = Field(..., gt=0, example=35.4)
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"name": "Foo", "price": 35.4}
            ]
        }
    }

app = FastAPI()

@app.post("/items/")
async def create_item(item: Item):
    return item
```

### 1.3 枚舉類型與 OpenAPI

Python 的枚舉類型自動轉換為 OpenAPI 中的枚舉：

```python
from enum import Enum
from fastapi import FastAPI

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

app = FastAPI()

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    return {"model_name": model_name}
```

## 2. 高級類型約束

### 2.1 類型約束對比表

| 約束類型 | 說明 | FastAPI/Pydantic 實現 |
|---------|------|---------------------|
| anyOf | 符合多個 schema 中的至少一個 | `Union[Type1, Type2]` |
| oneOf | 符合多個 schema 中的恰好一個 | 使用鑑別字段 (`discriminator`) |
| allOf | 同時符合所有指定的 schema | 類繼承 (Class inheritance) |
| not | 不符合指定的 schema | 自定義驗證器 (`field_validator`) |

### 2.2 使用 anyOf (Union)

```python
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

class StringModel(BaseModel):
    value: str

class NumberModel(BaseModel):
    value: float

app = FastAPI()

@app.post("/items/")
async def create_item(item: Union[StringModel, NumberModel]):
    return item
```

### 2.3 使用 oneOf (鑑別字段)

```python
from typing import Literal, Union
from fastapi import FastAPI
from pydantic import BaseModel, Field

class Dog(BaseModel):
    pet_type: Literal["dog"]
    bark: bool = True

class Cat(BaseModel):
    pet_type: Literal["cat"]
    meow: bool = True

class Pet(BaseModel):
    pet: Union[Dog, Cat] = Field(..., discriminator="pet_type")

app = FastAPI()

@app.post("/pets/")
async def create_pet(pet_info: Pet):
    return pet_info
```

### 2.4 使用 allOf (繼承)

```python
from fastapi import FastAPI
from pydantic import BaseModel

class NameModel(BaseModel):
    name: str

class AgeModel(BaseModel):
    age: int

class Person(NameModel, AgeModel):
    pass

app = FastAPI()

@app.post("/persons/")
async def create_person(person: Person):
    return person
```

### 2.5 使用 not (排除)

```python
from fastapi import FastAPI
from pydantic import BaseModel, field_validator

class User(BaseModel):
    username: str

    @field_validator('username')
    def username_not_reserved(cls, v):
        if v in ["admin", "root", "superuser"]:
            raise ValueError("保留用戶名")
        return v

app = FastAPI()

@app.post("/users/")
async def create_user(user: User):
    return user
```

## 3. 自定義 OpenAPI 文檔

### 3.1 文檔自定義選項

| 自定義項 | 說明 | 使用方式 |
|---------|------|---------|
| 操作 ID | 路徑操作的唯一標識符 | `@app.get(..., operation_id="get_item")` |
| 標籤 | 對端點進行分類分組 | `@app.get(..., tags=["users"])` |
| 排除端點 | 從文檔中隱藏端點 | `@app.get(..., include_in_schema=False)` |
| 自定義配置 | 添加額外的 OpenAPI 信息 | `@app.get(..., openapi_extra={...})` |

### 3.2 自定義操作 ID 與標籤

```python
from fastapi import FastAPI

app = FastAPI(
    openapi_tags=[
        {"name": "users", "description": "用戶相關操作"},
        {"name": "items", "description": "項目相關操作"},
    ]
)

@app.get("/users/", tags=["users"], operation_id="list_users")
async def read_users():
    return [{"name": "Harry"}, {"name": "Ron"}]
```

## 4. 高級響應處理

### 4.1 響應配置選項

| 配置項 | 說明 | 使用方式 |
|-------|------|---------|
| response_model | 響應數據模型 | `@app.get(..., response_model=Item)` |
| response_model_include | 只包含指定字段 | `@app.get(..., response_model_include={"name"})` |
| response_model_exclude | 排除指定字段 | `@app.get(..., response_model_exclude={"description"})` |
| responses | 定義不同狀態碼的響應 | `@app.get(..., responses={404: {"model": Error}})` |

### 4.2 動態響應模型

```python
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    id: str
    value: str

class Message(BaseModel):
    message: str

app = FastAPI()

@app.get("/items/{item_id}", response_model=Union[Item, Message])
async def read_item(item_id: str):
    if item_id == "foo":
        return {"id": "foo", "value": "bar"}
    return {"message": "Item not found"}
```

### 4.3 響應模型過濾

```python
from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str
    price: float
    tax: float

app = FastAPI()

@app.get(
    "/items/{item_id}",
    response_model=Item,
    response_model_include={"name", "price"}
)
async def read_item(item_id: str):
    return {
        "name": "Example",
        "description": "Full description",
        "price": 50.2,
        "tax": 10.5
    }
```

## 5. 安全性與認證

### 5.1 支持的安全方案

| 安全方案 | 說明 | FastAPI 類 |
|---------|------|-----------|
| OAuth2 密碼流 | 使用用戶名和密碼獲取令牌 | `OAuth2PasswordBearer` |
| OAuth2 客戶端憑證 | 使用客戶端 ID 和密鑰 | `OAuth2ClientCredentials` |
| API 密鑰 (Header) | 在標頭中傳遞 API 密鑰 | `APIKeyHeader` |
| API 密鑰 (Query) | 在查詢參數中傳遞 API 密鑰 | `APIKeyQuery` |
| API 密鑰 (Cookie) | 在 Cookie 中傳遞 API 密鑰 | `APIKeyCookie` |

### 5.2 OAuth2 基本示例

```python
from fastapi import Depends, FastAPI, Security
from fastapi.security import OAuth2PasswordBearer

app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.get("/items/")
async def read_items(token: str = Depends(oauth2_scheme)):
    return {"token": token}
```

### 5.3 API 密鑰示例

```python
from fastapi import Depends, FastAPI, Security
from fastapi.security import APIKeyHeader

app = FastAPI()

api_key_header = APIKeyHeader(name="X-API-Key")

@app.get("/items/")
async def read_items(api_key: str = Depends(api_key_header)):
    return {"api_key": api_key}
```

## 6. 依賴項與 OpenAPI

### 6.1 依賴項在 OpenAPI 中的表現

```python
from fastapi import Depends, FastAPI, Query

app = FastAPI()

async def common_params(
    q: str = Query(None, min_length=3),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, le=100)
):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items/")
async def read_items(commons: dict = Depends(common_params)):
    return commons
```

### 6.2 路徑依賴項

```python
from fastapi import Depends, FastAPI, Header, HTTPException

app = FastAPI()

async def verify_token(x_token: str = Header(...)):
    if x_token != "valid-token":
        raise HTTPException(status_code=400, detail="Invalid token")
    return x_token

@app.get("/items/", dependencies=[Depends(verify_token)])
async def read_items():
    return [{"item": "Foo"}]
```

## 7. 實用案例

### 7.1 多態性 API 端點

```python
from typing import Literal, Union
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

class BasePet(BaseModel):
    pet_type: str
    name: str

class Dog(BasePet):
    pet_type: Literal["dog"]
    breed: str

class Cat(BasePet):
    pet_type: Literal["cat"]
    breed: str

class PetRegistration(BaseModel):
    owner_name: str
    pet: Union[Dog, Cat] = Field(..., discriminator="pet_type")

@app.post("/register/")
async def register_pet(registration: PetRegistration):
    return registration
```

### 7.2 複雜驗證規則

```python
from datetime import date
from fastapi import FastAPI
from pydantic import BaseModel, field_validator

app = FastAPI()

class BookingDate(BaseModel):
    check_in: date
    check_out: date
    
    @field_validator('check_out')
    def check_dates(cls, v, info):
        if 'check_in' in info.data and v <= info.data['check_in']:
            raise ValueError('退房日期必須晚於入住日期')
        return v

@app.post("/bookings/")
async def create_booking(booking: BookingDate):
    return booking
```

## 8. 常用 Schema 定制技巧

### 8.1 常用 Field 參數

| 參數 | 說明 | 示例 |
|-----|------|-----|
| `default` | 默認值 | `Field(default=0)` |
| `title` | 字段標題 | `Field(title="項目名稱")` |
| `description` | 字段描述 | `Field(description="項目的詳細描述")` |
| `example` | 示例值 | `Field(example="範例值")` |
| `gt/ge` | 大於/大於等於 | `Field(ge=0)` |
| `lt/le` | 小於/小於等於 | `Field(lt=100)` |
| `min_length/max_length` | 字符串長度限制 | `Field(min_length=3, max_length=50)` |
| `regex` | 正則表達式 | `Field(regex="^[a-z]+$")` |

### 8.2 自定義 OpenAPI 示例

```python
from fastapi import FastAPI, Body
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Foo",
                    "price": 35.4
                }
            ]
        }
    }

app = FastAPI()

@app.post("/items/")
async def create_item(
    item: Item = Body(
        ...,
        examples=[
            {
                "summary": "基本示例",
                "value": {
                    "name": "Foo",
                    "price": 35.4
                }
            },
            {
                "summary": "高價示例",
                "value": {
                    "name": "Bar",
                    "price": 135.4
                }
            }
        ]
    )
):
    return item
```

## 9. 總結

OpenAPI Schema 進階用法的主要優勢：

| 功能 | 優勢 |
|------|------|
| 複雜數據模型 | 支持嵌套、繼承和複雜關係 |
| 高級類型約束 | 提供靈活的數據驗證機制 |
| 自定義文檔 | 改善 API 文檔的組織和可讀性 |
| 高級響應處理 | 精確控制 API 輸出 |
| 安全性與認證 | 內建多種安全機制並反映在文檔中 |
| 依賴項整合 | 簡化代碼並自動生成文檔 |

這些進階功能使 FastAPI 成為構建複雜、高性能 API 的理想選擇，同時保持了出色的文檔和類型安全性。