import os
import json

def count_images(path):
    if not os.path.exists(path):
        return 0
    return len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

# Get classes from train
classes = sorted([d for d in os.listdir('data/train_final_v2') if os.path.isdir(f'data/train_final_v2/{d}')])

stats = {}
for cls in classes:
    train_count = count_images(f'data/train_final_v2/{cls}')
    val_count = count_images(f'data/val_final_v2/{cls}')
    test_count = count_images(f'data/test_final_v2/{cls}')
    stats[cls] = {'train': train_count, 'val': val_count, 'test': test_count}

print(f'=== FINAL V2 DATASET SUMMARY ===')
print(f'Total classes: {len(classes)}')
print()

# Print table
print('Class                      Train    Val     Test    Total')
print('-' * 55)
total_train = total_val = total_test = 0
for cls, counts in sorted(stats.items()):
    t, v, te = counts['train'], counts['val'], counts['test']
    total_train += t
    total_val += v
    total_test += te
    print(f'{cls:27} {t:6} {v:6} {te:6} {t+v+te:6}')

print('-' * 55)
print(f'{"TOTAL":27} {total_train:6} {total_val:6} {total_test:6} {total_train+total_val+total_test:6}')

# Check for any class < 100 images
print()
print('=== Classes below 100 images ===')
for cls, counts in stats.items():
    total = counts['train'] + counts['val'] + counts['test']
    if total < 100:
        print(f'{cls}: {total} images')

# Save mapping
mapping = {i: cls for i, cls in enumerate(classes)}
with open('models/breed_mapping_final_v2.json', 'w') as f:
    json.dump({'classes': mapping, 'stats': stats}, f, indent=2)

print(f'\nSaved mapping to models/breed_mapping_final_v2.json')