"""
Test marking functionality using the functions from p1.py
This demonstrates how to use grade_marking_item_xverify and compute_score_p1 with marking.
"""
import sys
import os

# Add the parent directory to path to import p1
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rm_p1_slime import grade_marking_item_xverify, compute_score_p1, Model_args

# Example test case
problem = "For what value of $B_0$ does the total current $I_J = 0$?"

model_solution = """
To find when $I_J = 0$, we need to solve the equation:
$$I_J = I_1 + I_2 = 0$$

From the Josephson junction equations, we have:
$$I_1 = I_c \\sin(\\phi_1)$$
$$I_2 = I_c \\sin(\\phi_2)$$

The phase difference is related to the magnetic field:
$$\\phi_1 - \\phi_2 = \\frac{2\\pi B_0 (d + 2\\lambda)Y}{\\Phi_0}$$

where $\\Phi_0 = \\frac{h}{2e}$ is the flux quantum.

For $I_J = 0$, we need $\\sin(\\phi_1) + \\sin(\\phi_2) = 0$.

After algebraic manipulation, we find:
$$B_0 = n \\frac{\\pi \\hbar}{e (d + 2\\lambda)Y}$$

where $n = \\pm 1, \\pm 2, \\dots$

Therefore, the answer is: $B_0 = n \\frac{\\pi \\hbar}{e (d + 2\\lambda)Y}$ where $n = \\pm 1, \\pm 2, ...$
"""

ground_truth = "$B_0 = n \\frac{\\pi \\hbar}{e (d + 2\\lambda)Y}$ where $n = \\pm 1, \\pm 2, ...$"

# Example marking items - can be a list of dictionaries
marking_items = [
    {
        "desc": "Award 2 pt for correctly identifying the condition $I_J = 0$ and setting up the equation.",
        "pt": 2.0
    },
    {
        "desc": "Award 3 pt for deriving the relationship between magnetic field and phase difference.",
        "pt": 3.0,
        "expressions": ["$\\phi_1 - \\phi_2 = \\frac{2\\pi B_0 (d + 2\\lambda)Y}{\\Phi_0}$", "$\\Phi_0 = \\frac{h}{2e}$"]
    },
    {
        "desc": "Award 2 pt for correctly solving for $B_0$ and identifying the quantization condition.",
        "pt": 2.0
    }
]

# Configure model args
model_args = Model_args(model_port=34812)
model_args.api_key = "none"
model_args.base_url = "http://10.102.247.32:34812/v1"
model_args.model_name = "gpt-oss-120b"
model_args.temperature = 0.6
model_args.max_tokens = 4096

print("=" * 80)
print("TEST 1: Testing individual marking items with grade_marking_item_xverify")
print("=" * 80)

# Test each marking item individually
for i, marking_item in enumerate(marking_items, 1):
    print(f"\n--- Marking Item {i} ---")
    print(f"Description: {marking_item['desc']}")
    print(f"Points: {marking_item['pt']}")
    
    try:
        is_satisfied, llm_response = grade_marking_item_xverify(
            model_output=model_solution,
            marking_item=marking_item,
            problem=problem,
            model_args=model_args
        )
        
        print(f"Result: {'Satisfied' if is_satisfied else 'Not Satisfied'}")
        print(f"LLM Response: {llm_response[:200]}..." if len(llm_response) > 200 else f"LLM Response: {llm_response}")
        if is_satisfied:
            print(f"Points awarded: {marking_item['pt']}")
        else:
            print("Points awarded: 0")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("TEST 2: Testing complete scoring with compute_score_p1 (including marking)")
print("=" * 80)

# Test with compute_score_p1 which handles both answer grading and marking
result = compute_score_p1(
    model_output=model_solution,
    label=ground_truth,
    points=[7.0],  # Total points for the answer
    question=problem,
    use_xverify=True,
    model_port=34812,
    marking=marking_items  # Pass marking items
)

print(f"\nScore: {result['score']}")
print(f"Point: {result['point']}")
print(f"Accuracy: {result['acc']}")
print(f"Scored by: {result['scored_by']}")
print(f"Extracted prediction: {result['extracted_pred']}")
print(f"Extracted ground truth: {result['extracted_gt']}")
print(f"Score (no xverify): {result['score_noxverify']}")
print(f"Point (no xverify): {result['point_noxverify']}")

print("\n" + "=" * 80)
print("TEST 3: Testing with nested marking format (list of lists)")
print("=" * 80)

# Marking can also be in nested format: [[marking_item1, marking_item2], [marking_item3]]
nested_marking = [
    [
        {"desc": "Award 2 pt for correctly identifying the condition $I_J = 0$ and setting up the equation.", "pt": 2.0},
        {"desc": "Award 3 pt for deriving the relationship between magnetic field and phase difference.", "pt": 3.0}
    ],
    [
        {"desc": "Award 2 pt for correctly solving for $B_0$ and identifying the quantization condition.", "pt": 2.0}
    ]
]

result2 = compute_score_p1(
    model_output=model_solution,
    label=ground_truth,
    points=[7.0],
    question=problem,
    use_xverify=True,
    model_port=34812,
    marking=nested_marking
)

print(f"\nScore: {result2['score']}")
print(f"Point: {result2['point']}")
print(f"Accuracy: {result2['acc']}")

print("\n" + "=" * 80)

