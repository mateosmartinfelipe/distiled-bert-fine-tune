## Fine tuning a the distiled-bert model for sentimen analysis
 distiled-bert-fine-tune

### application schema on mermaid Data transformation
```mermaid
flowchart 
    A(web/url/data) --> B{Existing datasets} 
    B --> |No| C(Download zip)
    C -->  E(unzip to local)
    E -->  G(local to df)
    B --> |YES| G
    G --> H(df to tensor)
    H --> Dataset
    Dataset --> Batch;
```
### application schema on mermaid Data transformation
```mermaid
sequenceDiagram 
    autonumber
    participant Client
    participant OAthClient
    participant Service
    Client->>OAthClient: Request Token
    activate OAthClient
    OAthClient->>Client: send token
    deactivate OAthClient
    Client->>Service: request resource
    activate Service
    Service->>OAthClient: Verify token
    activate OAthClient
    OAthClient->>Service: Token verify
    deactivate OAthClient
    Service->>Client: Resource send
    deactivate Service
```
### class diagram
```mermaid
classDiagram
    class Order{
        +OrderStatus status
    }
    class OrderStatus{
        <<enumeration>>
        PAID
        ORDER
        STATUS
    }
    class PaymentProcessor{
        <<interface>>
        -String apiKey
        #connect(String url)
        +processPayment(Order order)OrderStatus
    }
    class Customer{
        +String name
    }
    Customer <|--PrivateCustomer 
    Customer <|--BusinessCustomer 
    PaymentProcessor <|-- Stripe
    PaymentProcessor <|-- Paypal
    Order o-- Customer
```
### entity relation diagram
```mermaid
erDiagram
    Customer ||--o{Order: places
    Order ||--|{LineItems:contains
    Customer{
        String id
        String name
    }
    Order{
        String id
        String orderStatus
    }
```