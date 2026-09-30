-  **VQL** is Denodo's language not SQL so a correct SQL might not work.
- To use four distinct models, for example one for **selection**, one for **generation**, one for **repair**, and one for **answering**, you would need to modify the SDK code or build your own orchestration around its endpoints.
- We should include **sample rows** in the prompt so it can know how the values of the columns actually look like.
- **Associations** might give us a big help. They connect related views together. For example, an order's `region_id` may match an ID in a regions view. A **join** is the query operation that combines matching rows to attach the region name. The relationship helps the model write the join correctly.
- **Metadata search** searches for relevant views, Sample search searches for relevant columns in these views. So its super helpful for finding specific columns in view(s) who have a lot of columns.

|Index|Contains|Helps answer|
|---|---|---|
|Metadata|View names, columns, descriptions, relationships|“Which views could answer this question?”|
|Sample data|Example records from those views|“What do the values look like?”|

|Search|What it selects|
|---|---|
|Metadata search|Relevant views|
|Sample search|Relevant example rows **within those views**|

- We could spin up agents to see if the views are relevant to the question of the user or not. This would tell us if the semantic layer is broken.
	- we should test out the semantic search does it work well in the first place?

- Set database/tag filters to narrow which views the SDK searches. We could use a drop-down list that the user can select views that they think related.

| Filter               | Meaning                                                   | Example     |
| -------------------- | --------------------------------------------------------- | ----------- |
| `vdp_database_names` | Limit the search to named Denodo databases.               | `"sales"`   |
| `vdp_tag_names`      | Limit the search to views carrying specified Denodo tags. | `"finance"` |

- We could make the agent use mode="metadata" so it can get more info about the metadata then the agent generates the VQL.