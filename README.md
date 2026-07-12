Quantum Resource Allocation Optimizer

A desktop application that uses quantum computing principles to optimize project resource allocation across tasks. Built with Python, Cirq, and Tkinter.

Note: This is a quantum-inspired constraint-encoding experiment, not an optimizer. It uses entanglement (CNOTs) to encode compatibility constraints, skill matching, timeline fit, probabilistically, and runs on Cirq's classical simulator (no physical quantum hardware). It does not define or minimize a cost function, so it is not performing optimization in the formal sense; final allocation weighting is done via classical post-processing (bandwidth/priority scaling). A variational approach (e.g. QAOA) with an explicit cost function is the natural next step and is not yet implemented.

The Problem It Solves

Project managers often struggle to allocate the right people to the right tasks — especially when multiple projects compete for the same resources. Traditional approaches rely on manual judgment or basic spreadsheet models. This tool uses a quantum-inspired approach to evaluate all possible resource-task combinations simultaneously and recommend optimal assignments based on skills, bandwidth, priority, and time constraints.

How It Works

The optimizer encodes each resource and task as a qubit. It then applies:


Hadamard gates to initialize all resources and tasks into superposition, representing all possible assignment combinations at once
Ry rotations to encode each resource's available bandwidth as a probability amplitude
Rz rotations to encode task priority into phase angles
CNOT gates to apply skill matching and time-based constraints between resource-task pairs
ZPowGate to encode timeline constraints based on task duration relative to project deadline


The circuit is run multiple times using Cirq's simulator to build a probability distribution. Higher measurement frequency for a resource-task pair indicates a stronger assignment recommendation.

Output

The application generates:


Resource to task assignment matrix with strength scores (0 to 1)
Bandwidth allocation per resource showing recommended hours per day per task
Gantt chart view of task assignments across the project timeline
Pie charts showing bandwidth split per resource
Interpretation guide categorizing assignments as Primary, Shared, Support, or Minimal


Tech Stack


Python 3.x
Cirq (quantum circuit simulation)
NumPy / Pandas
Matplotlib / Seaborn
Tkinter (desktop UI)
tkcalendar


How To Run


Install dependencies:


pip install cirq numpy pandas matplotlib seaborn tkcalendar


Run the application:


python P2_Enhanced.py


Enter project details, add resources with their skills and bandwidth availability, add tasks with required skills and priorities, then run the optimizer.


Background

This project was built as part of an internal innovation initiative at Mastech InfoTrellis and presented at an internal hackathon. It was later demonstrated to client leadership as a proof of concept for AI-assisted project planning.

The author has a background in quantum computing applied to resource optimization problems and holds an M.Tech in Computer Science with published research in machine intelligence (BIT Mesra, 2015).

License

Apache License 2.0
