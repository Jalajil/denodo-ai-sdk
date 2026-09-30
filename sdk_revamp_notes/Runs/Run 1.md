- Enable Reviewer LLM and make it be able to retry 3 times.
- Increase the `vector_search_sample_data_k` from 3 to 8. And preserve the row relationships, and empty values can remain as placeholders without breaking that relationship.
- Change comma-based serialization to be JSON instead, to preserve values that have commas. Apply the same as what is applied in ```C:\Users\gaming\repos\denodo-ai-sdk-enh.``` Change how sampled rows are stored, embedded, and read.
- set "mode" to "data"
- Set "verbose" to false.
- Set Qdrant and LLMs that we are using

Inspect the existing implementation first, then implement and verify each stage.

next I will check how prompts are written and check if we should use a specific model for a specific job

C:/Users/gaming/repos/denodo-ai-sdk/api/prompts/response/answer_view.py