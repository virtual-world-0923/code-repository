from dataclasses import dataclass


# @dataclass
class _calling_model_required:
    tokenizer = None

a = _calling_model_required()
a.tokenizer=1

print()

print(id(_calling_model_required()))
print(id(_calling_model_required().tokenizer))
print(id(_calling_model_required))
print(id(_calling_model_required.tokenizer))
print(id(a))

