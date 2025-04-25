from transformers import TrainingArguments

print("🧪 TrainingArguments version check:")
print("📦 Модуль:", TrainingArguments.__module__)
print("📄 Файл:", TrainingArguments.__init__.__code__.co_filename)
print("🔤 Аргументы конструктора:", TrainingArguments.__init__.__code__.co_varnames)
