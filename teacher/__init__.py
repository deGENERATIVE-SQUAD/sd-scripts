"""
SDXL Teacher-Student Training Module

This module implements teacher-student training for SDXL models based on the DMD2 distillation method.
The module provides a complete pipeline for generating teacher outputs and training student models.

Main components:
- generate_teacher_outputs.py: Generate teacher outputs from dataset
- train_student.py: Train student model using teacher outputs
- run_teacher_student.py: Complete pipeline runner
- quick_start.py: Interactive setup and quick start
- test_compatibility.py: Compatibility testing

For more information, see README.md or OVERVIEW.md
"""

__version__ = "1.0.0"
__author__ = "SD Scripts Community"
__description__ = "SDXL Teacher-Student Training Module"

# Import main classes for easy access
try:
    from .generate_teacher_outputs import TeacherOutputGenerator
    from .train_student import StudentTrainer, TeacherStudentDataset
except ImportError:
    # Allow import even if dependencies are not available
    pass

__all__ = [
    "TeacherOutputGenerator",
    "StudentTrainer", 
    "TeacherStudentDataset"
]