from dataclasses import dataclass

@dataclass
class Dataconfig:
    name: str = 'ergou'
    age: int = 4
    gender: str = 'male'

config = Dataconfig()
print(config.name)
print(config.age)
print(config.gender)