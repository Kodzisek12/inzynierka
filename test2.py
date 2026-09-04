import os

count = 0

for class_name in os.listdir("dataset_split/val"):
    class_dir = os.path.join("dataset_split/val", class_name)
    if os.path.isdir(class_dir):
        n = len([f for f in os.listdir(class_dir) if f.endswith(".avi")])
        print(class_name, n)
        count += n

print("Łącznie:", count)
count1=0
for class_name in os.listdir("dataset/UCF101"):
    class_dir = os.path.join("dataset/UCF101", class_name)
    if os.path.isdir(class_dir):
        count1 += len([f for f in os.listdir(class_dir) if f.endswith(".avi")])

print(count1)