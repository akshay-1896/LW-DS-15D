students_data = []   # List -> one dictionary per student
subjects = ('Math', 'Science', 'English')   # Tuple -> fixed list of subjects
used_rolls = set()  # Set -> keeps roll numbers unique

FILE_NAME = 'students_data.txt' # Constant -> holding the file name for entire program

def calculate_grade(average, passing_marks=40):
    """Decides the grade using conditional statements.\n
    'passing_marks' is a DEFAULT argument"""
    if average >= 95:
        grade = "A+"
    elif average >= 85:
        grade = "A"
    elif average >= 75:
        grade = "B"
    elif average >= 65:
        grade = "C"
    elif average >= 55:
        grade = "D"
    elif average >= passing_marks:
        grade = "E"
    else:
        grade = "F"
    return grade

def add_student():
    """Takes student details from the user, validates them, and stores them."""
    try:
        roll_no = int(input("Enter roll number: ")) # type conversion -> may raise ValueError

        if roll_no in used_rolls:
            raise Exception("This roll number is already used!")    # raise keyword

        name = input("Enter student name: ")

        marks = {}  # empty dictionary to store subject-wise marks
        total = 0

        for subject in subjects:    # for loop over a tuple
            score = int(input("Enter marks in {subject}(0-100): "))
            if score < 0 or score > 100:
                raise ValueError(f"Marks for {subject} must be between 0 and 100")
            marks[subject] = score
            total += score

    except ValueError as ve:
        print("Invalid input:", ve)
        return
    except Exception as e:
        print("Error:", e)
        return
    else:
        average = total / len(subjects)
        grade = calculate_grade(average,passing_marks=35)

        student = {
            'roll' : roll_no,
            'name' : name,
            'marks' : marks,
            'total' : total,
            'average' : average,
            'grade' : grade
        }

    students_data.append(student)
    used_rolls.add(roll_no)
    print(f'\nStudent {name} added successfully! Grade: {grade}')
    

    



























