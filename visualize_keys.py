import matplotlib.pyplot as plt
from collections import Counter

# Musical key distribution data
key_counts = Counter({
    'C:maj': 69, 'G:maj': 63, 'F:maj': 58, 'E:min': 56, 'Db:maj': 54,
    'D:maj': 52, 'Eb:maj': 51, 'Ab:maj': 51, 'C:min': 51, 'E:maj': 50,
    'A:maj': 49, 'A:min': 48, 'Bb:maj': 46, 'B:maj': 46, 'B:min': 45,
    'Gb:maj': 41, 'G:min': 38, 'Ab:min': 38, 'F:min': 37, 'D:min': 37,
    'Bb:min': 35, 'Db:min': 32, 'Eb:min': 32, 'Gb:min': 28
})

# Prepare data for pie chart
keys = list(key_counts.keys())
counts = list(key_counts.values())

# Create figure with larger size for better visibility
plt.figure(figsize=(14, 10))

# Create pie chart
colors = plt.cm.Set3(range(len(keys)))
wedges, texts, autotexts = plt.pie(
    counts, 
    labels=keys,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    textprops={'fontsize': 9}
)

# Enhance the percentage text
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(8)

plt.title('Distribution of Musical Keys in Dataset', fontsize=16, fontweight='bold', pad=20)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.tight_layout()

# Save the figure
plt.savefig('key_distribution.png', dpi=300, bbox_inches='tight')
print("Pie chart saved as 'key_distribution.png'")

# Display the chart
plt.show()

# Print summary statistics
print(f"\nTotal number of songs: {sum(counts)}")
print(f"Number of different keys: {len(keys)}")
print(f"\nTop 5 most common keys:")
for key, count in key_counts.most_common(5):
    percentage = (count / sum(counts)) * 100
    print(f"  {key}: {count} ({percentage:.1f}%)")
