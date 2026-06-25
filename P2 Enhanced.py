import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import datetime
from tkcalendar import DateEntry
import cirq
import warnings
from datetime import timedelta
import math
import logging
from typing import Dict, List, Any, Optional
from matplotlib.dates import DateFormatter, date2num

# Configure logging and suppress warnings
logging.basicConfig(
    filename='quantum_pm.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
warnings.filterwarnings('ignore')


class ErrorHandler:
    """Base class for error handling"""

    @staticmethod
    def handle_error(error_type: str, message: str, show_message: bool = True) -> None:
        """Generic error handler with logging"""
        error_log = f"{datetime.datetime.now()}: {error_type} - {message}"
        logging.error(error_log)

        if show_message:
            messagebox.showerror("Error", message)

    @staticmethod
    def validate_numeric_input(value: str, field_name: str,
                               min_value: Optional[float] = None,
                               max_value: Optional[float] = None) -> float:
        """Validate numeric inputs with range checking"""
        try:
            num_value = float(value)
            if min_value is not None and num_value < min_value:
                raise ValueError(f"{field_name} must be at least {min_value}")
            if max_value is not None and num_value > max_value:
                raise ValueError(f"{field_name} must be no more than {max_value}")
            return num_value
        except ValueError:
            raise ValueError(f"{field_name} must be a valid number")


class QuantumOptimizer:
    """Handle quantum optimization with error recovery"""

    def __init__(self, resources: List[Dict[str, Any]],
                 tasks: List[Dict[str, Any]],
                 project_details: Dict[str, Any]):
        self.resources = resources
        self.tasks = tasks
        self.project_details = project_details
        self.n_resources = len(resources)
        self.n_tasks = len(tasks)
        self.circuit = None
        self.results = None

    def create_quantum_circuit(self) -> cirq.Circuit:
        """Create quantum circuit with error handling"""
        try:
            # Create qubits
            resource_qubits = [cirq.GridQubit(i, 0) for i in range(self.n_resources)]
            task_qubits = [cirq.GridQubit(0, j) for j in range(self.n_tasks)]

            # Create constraint qubits
            bandwidth_qubits = [cirq.GridQubit(i, self.n_tasks + 1)
                                for i in range(self.n_resources)]
            priority_qubits = [cirq.GridQubit(self.n_resources + 1, j)
                               for j in range(self.n_tasks)]

            circuit = cirq.Circuit()

            # Initialize superposition
            circuit.append(cirq.H.on_each(resource_qubits))
            circuit.append(cirq.H.on_each(task_qubits))

            # Apply bandwidth constraints
            for i, resource in enumerate(self.resources):
                bandwidth = float(resource['bandwidth']) / 100.0
                theta = np.arccos(np.sqrt(bandwidth))
                circuit.append(cirq.Ry(theta).on(bandwidth_qubits[i]))
                circuit.append(cirq.CNOT(resource_qubits[i], bandwidth_qubits[i]))

            # Apply priority constraints
            for j, task in enumerate(self.tasks):
                priority_phase = (6 - task['priority']) / 5.0
                circuit.append(cirq.Rz(priority_phase * np.pi).on(priority_qubits[j]))
                circuit.append(cirq.CNOT(task_qubits[j], priority_qubits[j]))

            # Apply skill matching and time constraints
            self.apply_constraints(circuit, resource_qubits, task_qubits)

            # Measure results
            circuit.append(cirq.measure(*resource_qubits, key='resources'))
            circuit.append(cirq.measure(*task_qubits, key='tasks'))

            self.circuit = circuit
            return circuit

        except Exception as e:
            logging.error(f"Circuit creation failed: {str(e)}")
            raise

    def apply_constraints(self, circuit: cirq.Circuit,
                          resource_qubits: List[cirq.GridQubit],
                          task_qubits: List[cirq.GridQubit]) -> None:
        """Apply quantum constraints with error handling"""
        try:
            # Apply skill matching constraints
            for i, resource in enumerate(self.resources):
                for j, task in enumerate(self.tasks):
                    if self.check_skill_match(resource, task):
                        circuit.append(cirq.CNOT(resource_qubits[i], task_qubits[j]))

            # Apply time constraints
            self.apply_time_constraints(circuit, resource_qubits, task_qubits)

        except Exception as e:
            logging.error(f"Constraint application failed: {str(e)}")
            raise

    def apply_time_constraints(self, circuit: cirq.Circuit,
                               resource_qubits: List[cirq.GridQubit],
                               task_qubits: List[cirq.GridQubit]) -> None:
        """Apply time-based constraints with validation"""
        try:
            project_start = datetime.datetime.strptime(
                self.project_details['start_date'], '%Y-%m-%d').date()
            project_end = datetime.datetime.strptime(
                self.project_details['deadline'], '%Y-%m-%d').date()

            for i, resource in enumerate(self.resources):
                for j, task in enumerate(self.tasks):
                    task_start = datetime.datetime.strptime(
                        task['start_date'], '%Y-%m-%d').date()
                    task_end = datetime.datetime.strptime(
                        task['deadline'], '%Y-%m-%d').date()

                    if (task_start >= project_start and task_end <= project_end):
                        days_total = (project_end - project_start).days
                        days_task = (task_end - task_start).days
                        phase = (days_task / days_total) * np.pi

                        circuit.append(cirq.ZPowGate(exponent=phase).on(task_qubits[j]))
                        circuit.append(cirq.CNOT(resource_qubits[i], task_qubits[j]))

        except Exception as e:
            logging.error(f"Time constraint application failed: {str(e)}")
            raise

    def check_skill_match(self, resource: Dict[str, Any],
                          task: Dict[str, Any]) -> bool:
        """Check skill matching with error handling"""
        try:
            required_skills = set(task['required_skills'])
            primary_skills = set(resource['primary_skills'])
            secondary_skills = set(resource['secondary_skills'])

            # Primary skills match gives higher weight
            if required_skills & primary_skills:
                return True
            # Secondary skills match gives lower weight
            if required_skills & secondary_skills:
                return True
            return False

        except Exception as e:
            logging.error(f"Skill matching failed: {str(e)}")
            return False

    def run_optimization(self) -> np.ndarray:
        """Run quantum optimization with error recovery"""
        try:
            if self.circuit is None:
                self.create_quantum_circuit()

            simulator = cirq.Simulator()

            # Run multiple times to get distribution
            self.results = simulator.run(self.circuit, repetitions=1000)

            # Process results
            assignment_matrix = self.process_results()
            return self.post_process_assignments(assignment_matrix)

        except Exception as e:
            logging.error(f"Optimization failed: {str(e)}")
            # Return fallback solution
            return self.generate_fallback_solution()

    def process_results(self) -> np.ndarray:
        """Process quantum results with error handling"""
        try:
            measurements = self.results.measurements
            resource_states = measurements['resources']
            task_states = measurements['tasks']

            assignment_matrix = np.zeros((self.n_resources, self.n_tasks))

            for i in range(self.n_resources):
                for j in range(self.n_tasks):
                    assignment_matrix[i][j] = np.mean(
                        resource_states[:, i] & task_states[:, j]
                    )

            return assignment_matrix

        except Exception as e:
            logging.error(f"Result processing failed: {str(e)}")
            raise

    def post_process_assignments(self, assignment_matrix: np.ndarray) -> np.ndarray:
        """Apply post-processing with validation"""
        try:
            # Apply bandwidth constraints
            for i, resource in enumerate(self.resources):
                bandwidth = float(resource['bandwidth']) / 100.0
                assignment_matrix[i] *= bandwidth

            # Apply priority weighting
            for j, task in enumerate(self.tasks):
                priority_weight = (6 - task['priority']) / 5.0
                assignment_matrix[:, j] *= priority_weight

            # Normalize assignments
            row_sums = assignment_matrix.sum(axis=1)
            for i in range(self.n_resources):
                if row_sums[i] > 0:
                    assignment_matrix[i] /= row_sums[i]

            return assignment_matrix

        except Exception as e:
            logging.error(f"Post-processing failed: {str(e)}")
            raise

    def generate_fallback_solution(self) -> np.ndarray:
        """Generate fallback solution when optimization fails"""
        try:
            # Create basic assignment matrix
            matrix = np.zeros((self.n_resources, self.n_tasks))

            # Assign tasks based on priority and skills
            for j, task in enumerate(self.tasks):
                for i, resource in enumerate(self.resources):
                    if self.check_skill_match(resource, task):
                        matrix[i][j] = 1.0

            # Normalize assignments
            row_sums = matrix.sum(axis=1)
            for i in range(self.n_resources):
                if row_sums[i] > 0:
                    matrix[i] /= row_sums[i]

            return matrix

        except Exception as e:
            logging.error(f"Fallback solution generation failed: {str(e)}")
            return np.zeros((self.n_resources, self.n_tasks))


class ProjectManagerQuantumAssistant(ErrorHandler):
    def __init__(self, root: tk.Tk):
        """Initialize the application with error handling"""
        try:
            super().__init__()
            self.root = root
            self.root.title("Quantum Project Resource Optimizer")

            # Calculate window size based on screen size
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            window_width = int(screen_width * 0.8)
            window_height = int(screen_height * 0.8)

            # Center the window
            x = (screen_width - window_width) // 2
            y = (screen_height - window_height) // 2
            self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

            # Initialize data storage
            self.resources = []
            self.tasks = []
            self.project_details = {}

            # Predefined skillsets
            self.predefined_skills = [
                "Python", "Java", "JavaScript", "C++", "C#", "Ruby",
                "Selenium", "QA Automation", "Manual Testing",
                "AWS", "Azure", "GCP", "DevOps",
                "React", "Angular", "Vue.js",
                "Machine Learning", "AI", "Data Science",
                "Project Management", "Scrum Master",
                "Business Analysis", "System Architecture",
                "Database Administration", "SQL", "NoSQL",
                "Mobile Development", "iOS", "Android",
                "UI/UX Design", "Frontend", "Backend"
            ]

            # Configure styles
            self.configure_styles()

            # Initialize UI
            self.create_notebook()
            self.create_resource_tab()
            self.create_task_tab()
            self.create_project_tab()
            self.create_output_tab()

        except Exception as e:
            self.handle_error("Initialization Error", str(e))
            raise

    def configure_styles(self):
        """Configure ttk styles with error handling"""
        try:
            style = ttk.Style()

            # Configure notebook style
            style.configure('TNotebook', tabposition='n')
            style.configure('TNotebook.Tab', padding=[12, 4])

            # Configure entry styles
            style.configure('Valid.TEntry', fieldbackground='white')
            style.configure('Invalid.TEntry', fieldbackground='#FFE0E0')

            # Configure frame styles
            style.configure('Card.TFrame', relief='raised', padding=10)

            # Configure button styles
            style.configure('TButton', padding=[10, 5])

            # Configure label styles
            style.configure('TLabel', padding=[5, 5])

            # Configure treeview styles
            style.configure('Treeview', rowheight=25)
            style.configure('Treeview.Heading', font=('TkDefaultFont', 9, 'bold'))

        except Exception as e:
            self.handle_error("Style Configuration Error", str(e))

    def create_notebook(self):
        """Create main notebook with error handling"""
        try:
            self.notebook = ttk.Notebook(self.root)
            self.notebook.pack(pady=10, expand=True, fill='both')

            # Create tab frames
            self.resource_tab = ttk.Frame(self.notebook)
            self.task_tab = ttk.Frame(self.notebook)
            self.project_tab = ttk.Frame(self.notebook)
            self.output_tab = ttk.Frame(self.notebook)

            # Add tabs to notebook
            self.notebook.add(self.resource_tab, text="Resources")
            self.notebook.add(self.task_tab, text="Tasks")
            self.notebook.add(self.project_tab, text="Project Details")
            self.notebook.add(self.output_tab, text="Results")

        except Exception as e:
            self.handle_error("Notebook Creation Error", str(e))
            raise

    def create_resource_tab(self):
        """Create resource management tab with error handling"""
        try:
            # Create main frame with scrollbar
            main_frame = ttk.Frame(self.resource_tab)
            main_frame.pack(fill='both', expand=True, padx=10, pady=5)

            # Resource input form
            input_frame = ttk.LabelFrame(main_frame, text="Add Resource")
            input_frame.pack(fill='x', padx=5, pady=5)

            # Grid configuration
            for i in range(4):
                input_frame.grid_columnconfigure(i, weight=1)

            # Resource name
            ttk.Label(input_frame, text="Name:").grid(row=0, column=0, padx=5, pady=5)
            self.resource_name = ttk.Entry(input_frame)
            self.resource_name.grid(row=0, column=1, columnspan=3, sticky='ew', padx=5, pady=5)

            # Primary Skills
            ttk.Label(input_frame, text="Primary Skill:").grid(row=1, column=0, padx=5, pady=5)
            self.primary_skill = ttk.Combobox(input_frame, values=self.predefined_skills)
            self.primary_skill.grid(row=1, column=1, sticky='ew', padx=5, pady=5)

            # Secondary Skills
            ttk.Label(input_frame, text="Secondary Skill:").grid(row=1, column=2, padx=5, pady=5)
            self.secondary_skill = ttk.Combobox(input_frame, values=self.predefined_skills)
            self.secondary_skill.grid(row=1, column=3, sticky='ew', padx=5, pady=5)

            # Bandwidth
            ttk.Label(input_frame, text="Project Bandwidth (%):").grid(row=2, column=0, padx=5, pady=5)
            self.resource_bandwidth = ttk.Entry(input_frame)
            self.resource_bandwidth.grid(row=2, column=1, sticky='ew', padx=5, pady=5)

            # Experience
            ttk.Label(input_frame, text="Years of Experience:").grid(row=2, column=2, padx=5, pady=5)
            self.resource_experience = ttk.Entry(input_frame)
            self.resource_experience.grid(row=2, column=3, sticky='ew', padx=5, pady=5)

            # Leaves
            ttk.Label(input_frame, text="Planned Leaves:").grid(row=3, column=0, padx=5, pady=5)
            self.resource_leaves = ttk.Entry(input_frame)
            self.resource_leaves.grid(row=3, column=1, sticky='ew', padx=5, pady=5)

            ttk.Label(input_frame, text="Mandatory Leaves:").grid(row=3, column=2, padx=5, pady=5)
            self.resource_mandatory_leaves = ttk.Entry(input_frame)
            self.resource_mandatory_leaves.grid(row=3, column=3, sticky='ew', padx=5, pady=5)

            # Add resource button
            ttk.Button(input_frame, text="Add Resource",
                       command=self.add_resource).grid(row=4, column=0,
                                                       columnspan=4, pady=10)

            # Resource list
            list_frame = ttk.LabelFrame(main_frame, text="Resource List")
            list_frame.pack(fill='both', expand=True, padx=5, pady=5)

            # Create treeview with scrollbar
            columns = ("Name", "Primary Skill", "Secondary Skill", "Bandwidth",
                       "Experience", "Total Leaves")
            self.resource_tree = ttk.Treeview(list_frame, columns=columns,
                                              show='headings', height=10)

            # Configure columns
            for col in columns:
                self.resource_tree.heading(col, text=col)
                self.resource_tree.column(col, width=100)

            # Add scrollbar
            scrollbar = ttk.Scrollbar(list_frame, orient="vertical",
                                      command=self.resource_tree.yview)
            self.resource_tree.configure(yscrollcommand=scrollbar.set)

            # Pack list components
            self.resource_tree.pack(side='left', fill='both', expand=True)
            scrollbar.pack(side='right', fill='y')

        except Exception as e:
            self.handle_error("Resource Tab Creation Error", str(e))

    def add_resource(self):
        """Handle adding a new resource with validation"""
        try:
            # Validate inputs
            name = self.resource_name.get().strip()
            if not name:
                raise ValueError("Resource name is required")

            primary_skill = self.primary_skill.get()
            if not primary_skill:
                raise ValueError("Primary skill is required")

            secondary_skill = self.secondary_skill.get()
            if not secondary_skill:
                raise ValueError("Secondary skill is required")

            bandwidth = self.validate_numeric_input(
                self.resource_bandwidth.get().strip(), "Bandwidth", 0, 100)
            experience = self.validate_numeric_input(
                self.resource_experience.get().strip(), "Experience", 0)
            leaves = self.validate_numeric_input(
                self.resource_leaves.get().strip(), "Planned Leaves", 0)
            mandatory_leaves = self.validate_numeric_input(
                self.resource_mandatory_leaves.get().strip(), "Mandatory Leaves", 0)

            # Create resource dictionary
            resource = {
                'name': name,
                'primary_skills': [primary_skill],
                'secondary_skills': [secondary_skill],
                'bandwidth': bandwidth,
                'experience': experience,
                'leaves': leaves,
                'mandatory_leaves': mandatory_leaves
            }

            # Add to resources list
            self.resources.append(resource)

            # Add to treeview
            self.resource_tree.insert('', 'end', values=(
                name,
                primary_skill,
                secondary_skill,
                f"{bandwidth}%",
                f"{experience} years",
                f"{leaves + mandatory_leaves} days"
            ))

            # Clear input fields
            self.clear_resource_fields()

            messagebox.showinfo("Success", f"Resource {name} added successfully!")

        except ValueError as e:
            self.handle_error("Validation Error", str(e))
        except Exception as e:
            self.handle_error("Resource Addition Error", str(e))

    def clear_resource_fields(self):
        """Clear all resource input fields"""
        self.resource_name.delete(0, 'end')
        self.primary_skill.set('')
        self.secondary_skill.set('')
        self.resource_bandwidth.delete(0, 'end')
        self.resource_experience.delete(0, 'end')
        self.resource_leaves.delete(0, 'end')
        self.resource_mandatory_leaves.delete(0, 'end')

    def create_task_tab(self):
        """Create task management tab with error handling"""
        try:
            # Create main frame with scrollbar
            main_frame = ttk.Frame(self.task_tab)
            main_frame.pack(fill='both', expand=True, padx=10, pady=5)

            # Task input form
            input_frame = ttk.LabelFrame(main_frame, text="Add Task")
            input_frame.pack(fill='x', padx=5, pady=5)

            # Grid configuration
            for i in range(4):
                input_frame.grid_columnconfigure(i, weight=1)

            # Task name
            ttk.Label(input_frame, text="Task Name:").grid(row=0, column=0, padx=5, pady=5)
            self.task_name = ttk.Entry(input_frame)
            self.task_name.grid(row=0, column=1, columnspan=3, sticky='ew', padx=5, pady=5)

            # Required Skills
            ttk.Label(input_frame, text="Required Skill:").grid(row=1, column=0, padx=5, pady=5)
            self.task_skills = ttk.Combobox(input_frame, values=self.predefined_skills)
            self.task_skills.grid(row=1, column=1, sticky='ew', padx=5, pady=5)

            # Story Points
            ttk.Label(input_frame, text="Story Points:").grid(row=1, column=2, padx=5, pady=5)
            self.task_complexity = ttk.Combobox(input_frame, values=[1, 2, 3, 5, 8, 13, 21])
            self.task_complexity.grid(row=1, column=3, sticky='ew', padx=5, pady=5)

            # Priority
            ttk.Label(input_frame, text="Priority (1-5):").grid(row=2, column=0, padx=5, pady=5)
            self.task_priority = ttk.Combobox(input_frame, values=[1, 2, 3, 4, 5])
            self.task_priority.grid(row=2, column=1, sticky='ew', padx=5, pady=5)

            # Dates
            ttk.Label(input_frame, text="Start Date:").grid(row=2, column=2, padx=5, pady=5)
            self.task_start_date = DateEntry(input_frame, width=20)
            self.task_start_date.grid(row=2, column=3, sticky='ew', padx=5, pady=5)

            ttk.Label(input_frame, text="Deadline:").grid(row=3, column=0, padx=5, pady=5)
            self.task_deadline = DateEntry(input_frame, width=20)
            self.task_deadline.grid(row=3, column=1, sticky='ew', padx=5, pady=5)

            # Add task button
            ttk.Button(input_frame, text="Add Task",
                       command=self.add_task).grid(row=4, column=0,
                                                   columnspan=4, pady=10)

            # Task list
            list_frame = ttk.LabelFrame(main_frame, text="Task List")
            list_frame.pack(fill='both', expand=True, padx=5, pady=5)

            # Create treeview with scrollbar
            columns = ("Name", "Required Skills", "Story Points", "Priority",
                       "Start Date", "Deadline")
            self.task_tree = ttk.Treeview(list_frame, columns=columns,
                                          show='headings', height=10)

            # Configure columns
            for col in columns:
                self.task_tree.heading(col, text=col)
                self.task_tree.column(col, width=100)

            # Add scrollbar
            scrollbar = ttk.Scrollbar(list_frame, orient="vertical",
                                      command=self.task_tree.yview)
            self.task_tree.configure(yscrollcommand=scrollbar.set)

            # Pack list components
            self.task_tree.pack(side='left', fill='both', expand=True)
            scrollbar.pack(side='right', fill='y')

        except Exception as e:
            self.handle_error("Task Tab Creation Error", str(e))

    def add_task(self):
        """Handle adding a new task with validation"""
        try:
            # Validate inputs
            name = self.task_name.get().strip()
            if not name:
                raise ValueError("Task name is required")

            skills = self.task_skills.get()
            if not skills:
                raise ValueError("Required skills are required")

            try:
                complexity = int(self.task_complexity.get())
                priority = int(self.task_priority.get())
                if not 1 <= priority <= 5:
                    raise ValueError("Priority must be between 1 and 5")
            except:
                raise ValueError("Please select valid story points and priority")

            start_date = self.task_start_date.get_date()
            deadline = self.task_deadline.get_date()

            if start_date >= deadline:
                raise ValueError("Start date must be before deadline")

            # Create task dictionary
            task = {
                'name': name,
                'required_skills': [skills],
                'complexity': complexity,
                'priority': priority,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'deadline': deadline.strftime('%Y-%m-%d')
            }

            # Add to tasks list
            self.tasks.append(task)

            # Add to treeview
            self.task_tree.insert('', 'end', values=(
                name,
                skills,
                complexity,
                priority,
                start_date.strftime('%Y-%m-%d'),
                deadline.strftime('%Y-%m-%d')
            ))

            # Clear input fields
            self.clear_task_fields()

            messagebox.showinfo("Success", f"Task {name} added successfully!")

        except ValueError as e:
            self.handle_error("Validation Error", str(e))
        except Exception as e:
            self.handle_error("Task Addition Error", str(e))

    def clear_task_fields(self):
        """Clear all task input fields"""
        self.task_name.delete(0, 'end')
        self.task_skills.set('')
        self.task_complexity.set('')
        self.task_priority.set('')
        self.task_start_date.set_date(datetime.datetime.now())
        self.task_deadline.set_date(datetime.datetime.now())

    def create_project_tab(self):
        """Create project management tab with error handling"""
        try:
            project_frame = ttk.LabelFrame(self.project_tab, text="Project Details")
            project_frame.pack(padx=10, pady=5, fill='both', expand=True)

            # Grid configuration
            project_frame.grid_columnconfigure(1, weight=1)

            # Project name
            ttk.Label(project_frame, text="Project Name:").grid(row=0, column=0, padx=5, pady=5)
            self.project_name = ttk.Entry(project_frame)
            self.project_name.grid(row=0, column=1, sticky='ew', padx=5, pady=5)

            # Project dates
            ttk.Label(project_frame, text="Start Date:").grid(row=1, column=0, padx=5, pady=5)
            self.project_start_date = DateEntry(project_frame, width=20)
            self.project_start_date.grid(row=1, column=1, sticky='ew', padx=5, pady=5)

            ttk.Label(project_frame, text="Project Deadline:").grid(row=2, column=0, padx=5, pady=5)
            self.project_deadline = DateEntry(project_frame, width=20)
            self.project_deadline.grid(row=2, column=1, sticky='ew', padx=5, pady=5)

            # Optimization button
            ttk.Button(project_frame, text="Run Quantum Optimization",
                       command=self.run_optimization).grid(row=3, column=0,
                                                           columnspan=2, pady=20)

        except Exception as e:
            self.handle_error("Project Tab Creation Error", str(e))

    def create_output_tab(self):
        """Create results output tab with error handling"""
        try:
            # Create notebook for output views
            self.output_notebook = ttk.Notebook(self.output_tab)
            self.output_notebook.pack(expand=True, fill='both', padx=5, pady=5)

            # Create frames for different views
            self.matrix_frame = ttk.Frame(self.output_notebook)
            self.gantt_frame = ttk.Frame(self.output_notebook)
            self.calendar_frame = ttk.Frame(self.output_notebook)
            self.bandwidthdata_frame = ttk.Frame(self.output_notebook)
            self.bandwidth_frame = ttk.Frame(self.output_notebook)

            # Add frames to notebook
            self.output_notebook.add(self.matrix_frame, text="Resource Matrix")
            self.output_notebook.add(self.gantt_frame, text="Gantt Chart")
            self.output_notebook.add(self.calendar_frame, text="Calendar View")
            self.output_notebook.add(self.bandwidthdata_frame, text="Bandwidth Data")
            self.output_notebook.add(self.bandwidth_frame, text="Bandwidth Allocation")

        except Exception as e:
            self.handle_error("Output Tab Creation Error", str(e))

    def run_optimization(self):
        """Execute optimization and display results"""
        try:
            if not self.validate_inputs():
                return

            # Create quantum optimizer
            optimizer = QuantumOptimizer(self.resources, self.tasks, self.project_details)

            # Run optimization
            assignment_matrix = optimizer.run_optimization()

            # Display results
            self.display_resource_matrix(assignment_matrix)
            self.display_gantt_chart(assignment_matrix)
            self.display_calendar_view(assignment_matrix)
            self.calculate_bandwidth_allocation(assignment_matrix)
            self.display_bandwidth_graphs(assignment_matrix)

            # Switch to results tab
            self.notebook.select(self.output_tab)

            messagebox.showinfo("Success", "Optimization completed successfully!")

        except Exception as e:
            self.handle_error("Optimization Error", str(e))

    def validate_inputs(self):
        """Validate all inputs before optimization"""
        try:
            if not self.resources:
                raise ValueError("Please add at least one resource!")

            if not self.tasks:
                raise ValueError("Please add at least one task!")

            project_name = self.project_name.get().strip()
            if not project_name:
                raise ValueError("Please enter project name!")

            start_date = self.project_start_date.get_date()
            deadline = self.project_deadline.get_date()

            if start_date >= deadline:
                raise ValueError("Project start date must be before deadline!")

            self.project_details = {
                'name': project_name,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'deadline': deadline.strftime('%Y-%m-%d'),
                'working_days': self.calculate_working_days(start_date, deadline)
            }

            return True

        except ValueError as e:
            self.handle_error("Validation Error", str(e))
            return False
        except Exception as e:
            self.handle_error("Validation Error", str(e))
            return False

    def calculate_working_days(self, start_date: datetime.date,
                               end_date: datetime.date) -> int:
        """Calculate working days between dates"""
        days = 0
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:  # Monday = 0, Sunday = 6
                days += 1
            current += timedelta(days=1)
        return days

    def display_resource_matrix(self, assignment_matrix: np.ndarray):
        """Display resource allocation matrix with error handling"""
        try:
            # Clear previous content
            for widget in self.matrix_frame.winfo_children():
                widget.destroy()

            # Create container frame
            container = ttk.Frame(self.matrix_frame)
            container.pack(fill='both', expand=True, padx=5, pady=5)

            # Create DataFrame for visualization
            resource_names = [r['name'] for r in self.resources]
            task_names = [t['name'] for t in self.tasks]
            df = pd.DataFrame(assignment_matrix,
                              index=resource_names,
                              columns=task_names)

            # Calculate figure size based on number of resources and tasks
            width = max(12, len(task_names) * 1.2)  # Minimum width of 12, or 1.2 inches per task
            height = max(8, len(resource_names) * 0.8)  # Minimum height of 8, or 0.8 inches per resource

            # Create figure and axis
            fig = Figure(figsize=(width, height))
            ax = fig.add_subplot(111)

            # Create heatmap with adjusted font sizes
            heatmap = sns.heatmap(df, annot=True, cmap='YlOrRd', ax=ax, fmt='.2f',
                                  cbar_kws={'label': 'Assignment Strength'},
                                  annot_kws={'size': 8})  # Smaller font for numbers

            # Customize chart
            ax.set_title("Resource-Task Assignment Matrix", pad=20)  # Add padding to title

            # Rotate x-axis labels and adjust their size
            ax.set_xticklabels(ax.get_xticklabels(),
                               rotation=45,
                               ha='right',
                               fontsize=8)  # Smaller font for task names

            # Adjust y-axis labels size
            ax.set_yticklabels(ax.get_yticklabels(),
                               fontsize=8)  # Smaller font for resource names

            # Add legend with smaller font
            legend_text = ("Assignment Strength:\n"
                           "0.0-0.3: Minimal involvement\n"
                           "0.3-0.7: Partial assignment\n"
                           "0.7-1.0: Primary assignment")

            # Position legend based on matrix size
            legend_x = 1.02  # Slightly further right
            legend_y = 0.5  # Middle height

            ax.text(legend_x, legend_y, legend_text,
                    transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8),
                    fontsize=8,  # Smaller font for legend
                    verticalalignment='center')

            # Adjust layout with more space for labels
            fig.tight_layout(rect=[0, 0, 0.85, 0.95])  # Leave space for legend

            # Create scrolled canvas
            canvas_frame = ttk.Frame(container)
            canvas_frame.pack(fill='both', expand=True)

            canvas = tk.Canvas(canvas_frame)
            scrollbar_y = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
            scrollbar_x = ttk.Scrollbar(canvas_frame, orient="horizontal", command=canvas.xview)

            # Configure canvas scrolling
            canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

            # Create canvas with scrollbar
            chart_canvas = FigureCanvasTkAgg(fig, canvas)
            chart_canvas.draw()

            # Pack scrollbars and canvas
            scrollbar_y.pack(side='right', fill='y')
            scrollbar_x.pack(side='bottom', fill='x')
            canvas.pack(side='left', fill='both', expand=True)

            # Add the figure to the canvas
            canvas.create_window((0, 0), window=chart_canvas.get_tk_widget(), anchor='nw')

            # Update scroll region
            canvas.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox('all'))

            # Add toolbar
            toolbar = NavigationToolbar2Tk(chart_canvas, container)
            toolbar.update()

            # Add mousewheel scrolling
            def _on_mousewheel(event):
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

            def _on_shift_mousewheel(event):
                canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Shift-MouseWheel>", _on_shift_mousewheel)

        except Exception as e:
            self.handle_error("Matrix Display Error", str(e))

    def display_gantt_chart(self, assignment_matrix: np.ndarray):
        """Display Gantt chart with error handling"""
        try:
            # Clear previous content
            for widget in self.gantt_frame.winfo_children():
                widget.destroy()

            # Create container frame
            container = ttk.Frame(self.gantt_frame)
            container.pack(fill='both', expand=True, padx=5, pady=5)

            # Create figure and axis
            fig = Figure(figsize=(12, 8))
            ax = fig.add_subplot(111)

            # Process data
            resource_names = [r['name'] for r in self.resources]
            task_names = [t['name'] for t in self.tasks]

            # Create task bars data
            y_positions = []
            task_starts = []
            task_durations = []
            task_labels = []
            colors = []

            for i, resource in enumerate(resource_names):
                for j, task in enumerate(task_names):
                    if assignment_matrix[i][j] > 0.3:  # Significant assignment
                        start_date = datetime.datetime.strptime(
                            self.tasks[j]['start_date'], '%Y-%m-%d')
                        end_date = datetime.datetime.strptime(
                            self.tasks[j]['deadline'], '%Y-%m-%d')

                        # Convert dates to numbers for plotting
                        start_num = date2num(start_date)
                        duration = date2num(end_date) - start_num

                        y_positions.append(i)
                        task_starts.append(start_num)
                        task_durations.append(duration)
                        task_labels.append(f"{task}\n({resource})")

                        # Color based on assignment strength
                        if assignment_matrix[i][j] > 0.7:
                            colors.append('#ff9999')  # Strong
                        elif assignment_matrix[i][j] > 0.5:
                            colors.append('#66b3ff')  # Medium
                        else:
                            colors.append('#99ff99')  # Light

            # Create Gantt chart
            bars = ax.barh(y_positions, task_durations, left=task_starts,
                           height=0.3, color=colors)

            # Customize chart
            ax.set_yticks(range(len(resource_names)))
            ax.set_yticklabels(resource_names)
            ax.set_title("Project Schedule Gantt Chart")

            # Format dates
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()

            # Add task labels
            for i, (bar, label) in enumerate(zip(bars, task_labels)):
                ax.text(bar.get_x(), y_positions[i], label,
                        va='center', ha='right', fontsize=8)

            # Add legend
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, facecolor='#ff9999', label='Primary (>70%)'),
                plt.Rectangle((0, 0), 1, 1, facecolor='#66b3ff', label='Shared (50-70%)'),
                plt.Rectangle((0, 0), 1, 1, facecolor='#99ff99', label='Support (30-50%)')
            ]
            ax.legend(handles=legend_elements, loc='upper right')

            # Adjust layout
            fig.tight_layout()

            # Create canvas with scrollbar
            canvas = FigureCanvasTkAgg(fig, container)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)

            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, container)
            toolbar.update()

        except Exception as e:
            self.handle_error("Gantt Chart Display Error", str(e))

    def display_calendar_view(self, assignment_matrix: np.ndarray):
        """Display calendar view with error handling"""
        try:
            # Clear previous content
            for widget in self.calendar_frame.winfo_children():
                widget.destroy()

            # Create container frame
            container = ttk.Frame(self.calendar_frame)
            container.pack(fill='both', expand=True, padx=5, pady=5)

            # Create text widget with scrollbar
            text_widget = scrolledtext.ScrolledText(container,
                                                    wrap=tk.WORD,
                                                    width=80,
                                                    height=40)
            text_widget.pack(fill='both', expand=True)

            # Add calendar entries
            text_widget.insert(tk.END, "Resource Calendar View\n")
            text_widget.insert(tk.END, "=" * 80 + "\n\n")

            resource_names = [r['name'] for r in self.resources]
            task_names = [t['name'] for t in self.tasks]

            for i, resource in enumerate(resource_names):
                text_widget.insert(tk.END, f"\nResource: {resource}\n")
                text_widget.insert(tk.END, "-" * 40 + "\n")

                # Sort tasks by start date
                resource_tasks = []
                for j, task in enumerate(self.tasks):
                    if assignment_matrix[i][j] > 0.3:
                        resource_tasks.append({
                            'name': task['name'],
                            'start': datetime.datetime.strptime(
                                task['start_date'], '%Y-%m-%d').date(),
                            'end': datetime.datetime.strptime(
                                task['deadline'], '%Y-%m-%d').date(),
                            'strength': assignment_matrix[i][j],
                            'priority': task['priority']
                        })

                resource_tasks.sort(key=lambda x: x['start'])

                # Display tasks
                for task in resource_tasks:
                    assignment_type = ("Primary" if task['strength'] > 0.7 else
                                       "Shared" if task['strength'] > 0.5 else
                                       "Support")

                    text_widget.insert(tk.END,
                                       f"\nTask: {task['name']}\n"
                                       f"Role: {assignment_type}\n"
                                       f"Start: {task['start'].strftime('%Y-%m-%d')}\n"
                                       f"End: {task['end'].strftime('%Y-%m-%d')}\n"
                                       f"Priority: {task['priority']}\n"
                                       f"Assignment Strength: {task['strength']:.2f}\n"
                                       )

            # Make text widget read-only
            text_widget.configure(state='disabled')

        except Exception as e:
            self.handle_error("Calendar View Display Error", str(e))

    def calculate_bandwidth_allocation(self, assignment_matrix):
        """Calculate and display bandwidth allocation recommendations"""
        # Clear previous content
        if hasattr(self, 'bandwidth_frame'):
            for widget in self.bandwidthdata_frame.winfo_children():
                widget.destroy()
        else:
            self.bandwidthdata_frame = ttk.Frame(self.output_notebook)
            self.output_notebook.add(self.bandwidthdata_frame, text="Bandwidth Allocation")
        '''for widget in self.bandwidthdata_frame.winfo_children():
            widget.destroy()'''

        try:
            # Create container frame
            container = ttk.Frame(self.bandwidthdata_frame)
            container.pack(fill='both', expand=True, padx=5, pady=5)

            # Create text widget for detailed recommendations
            text_widget = tk.Text(self.bandwidthdata_frame, wrap=tk.WORD, height=20)
            text_widget.pack(fill='both', expand=True, padx=10, pady=5)

            resource_names = [r['name'] for r in self.resources]
            task_names = [t['name'] for t in self.tasks]

            text_widget.insert(tk.END, "Resource Bandwidth Allocation Recommendations\n")
            text_widget.insert(tk.END, "=" * 50 + "\n\n")

            # Calculate working hours per day (assuming 8-hour workday)
            HOURS_PER_DAY = 8

            for i, resource in enumerate(resource_names):
                text_widget.insert(tk.END, f"\nResource: {resource}\n")
                text_widget.insert(tk.END, "-" * 30 + "\n")

                # Calculate total assignment strength for normalization
                total_strength = sum(assignment_matrix[i])

                if total_strength > 0:
                    for j, task in enumerate(task_names):
                        strength = assignment_matrix[i][j]
                        if strength > 0:
                            # Calculate recommended hours based on assignment strength
                            if strength > 0.8:
                                recommended_hours = HOURS_PER_DAY * 0.8  # 80% of day
                                allocation_type = "Primary Task"
                            elif strength > 0.5:
                                recommended_hours = HOURS_PER_DAY * 0.5  # 50% of day
                                allocation_type = "Shared Task"
                            elif strength > 0.3:
                                recommended_hours = HOURS_PER_DAY * 0.3  # 30% of day
                                allocation_type = "Support Task"
                            else:
                                recommended_hours = HOURS_PER_DAY * 0.1  # 10% of day
                                allocation_type = "Minimal Support"

                            # Calculate days needed based on task complexity
                            story_points = self.tasks[j]['complexity']
                            estimated_days = math.ceil((story_points * 8) / recommended_hours)

                            text_widget.insert(tk.END,
                                               f"Task: {task}\n"
                                               f"Assignment Strength: {strength:.2f}\n"
                                               f"Allocation Type: {allocation_type}\n"
                                               f"Recommended Hours/Day: {recommended_hours:.1f}\n"
                                               f"Estimated Days: {estimated_days}\n"
                                               f"Story Points: {story_points}\n\n"
                                               )

            # Add example interpretations
            text_widget.insert(tk.END, "\nInterpretation Guide:\n")
            text_widget.insert(tk.END, "=" * 50 + "\n")
            text_widget.insert(tk.END, """
    Assignment Strength Interpretation:
    0.8 - 1.0 (Primary Task):
    - Dedicate 80% of daily bandwidth (6-7 hours/day)
    - Resource should be primary owner
    - Minimal interruptions recommended
    - Example: Core development tasks

    0.5 - 0.8 (Shared Task):
    - Dedicate 50% of daily bandwidth (4 hours/day)
    - Can be shared with one other major task
    - Regular context switching expected
    - Example: Development + Code Review

    0.3 - 0.5 (Support Task):
    - Dedicate 30% of daily bandwidth (2-3 hours/day)
    - Can be combined with other tasks
    - Intermittent focus needed
    - Example: Technical consultation, review tasks

    0.1 - 0.3 (Minimal Support):
    - Dedicate 10% of daily bandwidth (1 hour/day)
    - Ad-hoc support role
    - Can handle multiple such tasks
    - Example: Advisory role, lightweight reviews
    """)

            # Make text widget read-only
            text_widget.configure(state='disabled')

            # Add scrollbar
            scrollbar = ttk.Scrollbar(self.bandwidthdata_frame, orient='vertical',
                                      command=text_widget.yview)
            scrollbar.pack(side='right', fill='y')
            text_widget.configure(yscrollcommand=scrollbar.set)

        except Exception as e:
            messagebox.showerror("Error", f"Error calculating bandwidth allocation: {str(e)}")

    def display_bandwidth_graphs(self, assignment_matrix: np.ndarray):
        """Display bandwidth allocation graphs with error handling"""
        try:
            # Clear previous content
            for widget in self.bandwidth_frame.winfo_children():
                widget.destroy()

            # Create container frame with scrollbar
            container = ttk.Frame(self.bandwidth_frame)
            container.pack(fill='both', expand=True, padx=5, pady=5)

            canvas = tk.Canvas(container)
            scrollbar = ttk.Scrollbar(container, orient="vertical",
                                      command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)

            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            # Get data
            resource_names = [r['name'] for r in self.resources]
            task_names = [t['name'] for t in self.tasks]

            # Create figure with subplots
            n_resources = len(resource_names)
            fig_height = max(8, n_resources * 3)  # Adjust height based on number of resources
            fig = Figure(figsize=(12, fig_height))

            # Create pie charts for each resource
            for i, resource in enumerate(resource_names):
                ax = fig.add_subplot(n_resources, 1, i + 1)

                # Calculate allocations
                allocations = assignment_matrix[i]
                labels = []
                sizes = []
                colors = []

                for j, strength in enumerate(allocations):
                    if strength > 0.1:  # Show significant allocations
                        labels.append(f"{task_names[j]}\n({strength:.2f})")
                        sizes.append(strength)

                        if strength > 0.7:
                            colors.append('#ff9999')  # Strong
                        elif strength > 0.5:
                            colors.append('#66b3ff')  # Medium
                        else:
                            colors.append('#99ff99')  # Light

                # Add unallocated time
                total_allocation = sum(sizes)
                if total_allocation < 1:
                    labels.append('Unallocated')
                    sizes.append(1 - total_allocation)
                    colors.append('#dddddd')

                # Create pie chart
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
                ax.set_title(f"{resource} Bandwidth Allocation")

            # Adjust layout
            fig.tight_layout()

            # Create canvas
            chart_canvas = FigureCanvasTkAgg(fig, scrollable_frame)
            chart_canvas.draw()
            chart_canvas.get_tk_widget().pack(fill='both', expand=True)

            # Pack scrollbar components
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

        except Exception as e:
            self.handle_error("Bandwidth Graph Display Error", str(e))


def main():
    """Main application entry point with error handling"""
    try:
        root = tk.Tk()

        # Configure style
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')

        # Create and run application
        app = ProjectManagerQuantumAssistant(root)

        # Center window on screen
        window_width = int(root.winfo_screenwidth() * 0.8)
        window_height = int(root.winfo_screenheight() * 0.8)
        x = (root.winfo_screenwidth() - window_width) // 2
        y = (root.winfo_screenheight() - window_height) // 2
        root.geometry(f'{window_width}x{window_height}+{x}+{y}')

        root.mainloop()

    except Exception as e:
        logging.error(f"Application failed to start: {str(e)}")
        messagebox.showerror("Fatal Error",
                             f"Application failed to start: {str(e)}\n\n"
                             f"Please check the log file for details.")


if __name__ == "__main__":
    main()
