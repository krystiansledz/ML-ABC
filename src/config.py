class Config:
    fruits: list[str] = ["apple", "raspberry", "mango", "lemon"]
    data_path: str = "./data"
    output_path: str = "./out"
    bees_count: int = 5
    generations_count: int = 500
    mutation_probability: int = 5
    log_iterations: bool = True
    train_data_ratio: float = 0.7
    max_no_improvement_limit: int = 10
    min_value: int = -5
    max_value: int = 5
