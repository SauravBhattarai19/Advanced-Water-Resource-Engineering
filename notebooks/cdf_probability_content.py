# MARKDOWN AND CODE FOR JUPYTER NOTEBOOK
# Copy the content below into your .ipynb file

MARKDOWN_CONTENT = """
## 🌊 Return Period to Flow Calculator

This code calculates the **flow value** for a given **return period** using your best fitted distribution.

**Features:**
- Step-by-step calculation showing the math
- Interactive: Change the return period to see different results
- Visual: Plots the PPF curve to show the relationship

**Change the return period below to calculate different flood flows!**
"""

CODE_CONTENT = '''
# Step 1: Set your return period (in years)
return_period = 100  # Change this number! Try 50, 25, 10, etc.

# Step 2: Convert return period to probability
prob_exceedance = 1.0 / return_period  # Chance of flood each year
prob_non_exceedance = 1 - prob_exceedance  # Chance of NO flood each year

print(f"Return period: {return_period} years")
print(f"Probability of exceedance: {prob_exceedance:.3f} ({prob_exceedance*100:.1f}%)")
print(f"Probability of non-exceedance: {prob_non_exceedance:.3f} ({prob_non_exceedance*100:.1f}%)")
print()

# Step 3: Calculate flow value using inverse CDF (PPF)
if best_shape == 'gumbel':
    flow_value = stats.gumbel_r.ppf(prob_non_exceedance, gumbel_location, gumbel_scale)
    shape_name = 'Gumbel'
elif best_shape == 'lognorm':
    flow_value = stats.lognorm.ppf(prob_non_exceedance, lognorm_shape, loc=lognorm_location, scale=lognorm_scale)
    shape_name = 'Log-Normal'
else:  # normal
    flow_value = stats.norm.ppf(prob_non_exceedance, normal_average, normal_spread)
    shape_name = 'Normal'

# Step 4: Show the result
print("🎯 RESULT:")
print(f"Using {shape_name} distribution:")
print(f"A {return_period}-year flood has a flow of {flow_value:.0f} m³/s")
print(f"This means there's a {prob_exceedance*100:.1f}% chance each year of exceeding this flow")
print()

# Step 5: Plot the PPF curve to visualize the relationship
# Create range of probabilities
prob_range = np.linspace(0.001, 0.999, 100)  # Avoid 0 and 1 for numerical stability

# Calculate corresponding flow values using PPF
if best_shape == 'gumbel':
    flow_range = stats.gumbel_r.ppf(prob_range, gumbel_location, gumbel_scale)
elif best_shape == 'lognorm':
    flow_range = stats.lognorm.ppf(prob_range, lognorm_shape, loc=lognorm_location, scale=lognorm_scale)
else:  # normal
    flow_range = stats.norm.ppf(prob_range, normal_average, normal_spread)

# Create the plot
plt.figure(figsize=(12, 6))

# Plot PPF curve
plt.plot(prob_range, flow_range, 'b-', linewidth=2, label=f'{shape_name} PPF Curve')

# Mark our specific point
plt.plot(prob_non_exceedance, flow_value, 'ro', markersize=8, label=f'{return_period}-year flood')

# Add grid and labels
plt.xlabel('Non-exceedance Probability', fontsize=12)
plt.ylabel('Flow Value (m³/s)', fontsize=12)
plt.title(f'PPF Curve: Return Period to Flow Relationship ({shape_name})', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

# Add text annotation
plt.annotate(f'{return_period}-year flood\\n{flow_value:.0f} m³/s',
            xy=(prob_non_exceedance, flow_value),
            xytext=(prob_non_exceedance + 0.1, flow_value + 50),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')

plt.tight_layout()
plt.show()

print()
print("📊 The PPF curve shows:")
print("- X-axis: Non-exceedance probability (how often flow is NOT exceeded)")
print("- Y-axis: Flow value (m³/s)")
print(f"- Red dot: Your {return_period}-year flood calculation")
print("- Higher return periods = higher flow values = lower probabilities")
'''

# Print the formatted content for easy copying
print("="*60)
print("COPY THIS INTO YOUR JUPYTER NOTEBOOK:")
print("="*60)

print("\n" + MARKDOWN_CONTENT)

print("\n```python")
print(CODE_CONTENT.strip())
print("```")

print("\n" + "="*60)
print("INSTRUCTIONS:")
print("="*60)
print("1. Create a new Markdown cell in your notebook")
print("2. Copy the markdown text above into it")
print("3. Create a new Code cell below it")
print("4. Copy the Python code into the code cell")
print("5. Run the code cell to see the results")
