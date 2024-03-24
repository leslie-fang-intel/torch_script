from transformers import pipeline

def test_task_fill_mask():
    unmasker = pipeline('fill-mask', model='bert-large-uncased')
    res = unmasker("Artificial Intelligence [MASK] take over the world.")
    print("--------------", flush=True)
    print(res, flush=True)

if __name__ == "__main__":
    test_task_fill_mask()
