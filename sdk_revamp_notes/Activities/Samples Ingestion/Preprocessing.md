#### **comma-based serialization should be JSON instead.**
**Example:**
page_content: London, UK,active,00123
columns:      city,status,customer_id

Becomes:

| Column        | Original value | Reconstructed sample |
| ------------- | -------------- | -------------------- |
| `city`        | `London, UK`   | `London`             |
| `status`      | `active`       | `UK`                 |
| `customer_id` | `00123`        | `active`             |

