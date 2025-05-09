# 生成完整的 Python 脚本：从 CSV 导入、转换为 RDF、保存为 .ttl 文件
import os
import pandas as pd
from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef
from rdflib.namespace import XSD
from rdflib.namespace import OWL

g = Graph()
EX = Namespace("http://www.example.org/university#")
g.bind("ex", EX)

ontology_file = "Assignment2.ttl"
g.parse(ontology_file, format="turtle")

def uri(class_name, id_val):
    return URIRef(f"http://www.example.org/university#{class_name}_{str(id_val).replace(' ', '_')}")

# ======================
# 转换函数定义
# ======================

# 2) 定义 Departments & Divisions
def convert_departments_and_divisions(courses_df, senior_df, ta_df, programmes_df):
    # Department：Courses.csv, Senior_Teachers.csv , Teaching_Assistants.csv, Programmes.csv
    all_depts = set(courses_df['Department']) \
              | set(senior_df['Department name']) \
              | set(ta_df['Department name']) \
              | set(programmes_df['Department name'])

    # Division：Courses.csv， Senior_Teachers.csv， Teaching_Assistants.csv
    all_divs = set(courses_df['Division']) \
             | set(senior_df['Division name']) \
             | set(ta_df['Division name'])

    # Department 实例
    for dept in all_depts:
        d_uri = uri("Department", dept)
        g.add((d_uri, RDF.type, EX.Department))

        g.add((d_uri, EX.deptId, Literal(dept, datatype=XSD.string)))
        g.add((d_uri, EX.departmentName, Literal(dept, datatype=XSD.string)))

    # 创建所有 Division 实例，并关联到 Department
    for div in all_divs:
        dv_uri = uri("Division", div)
        g.add((dv_uri, RDF.type, EX.Division))
        g.add((dv_uri, EX.divisionName, Literal(div, datatype=XSD.string)))

        # 尝试从 Courses/Senior/TA 找到它对应的 department
        if div in courses_df['Division'].values:
            parent = courses_df.loc[courses_df['Division']==div, 'Department'].iloc[0]
        elif div in senior_df['Division name'].values:
            parent = senior_df.loc[senior_df['Division name']==div, 'Department name'].iloc[0]
        else:
            parent = ta_df.loc[ta_df['Division name']==div, 'Department name'].iloc[0]

        pd_uri = uri("Department", parent)
        g.add((dv_uri, EX.partOfDepartment, pd_uri))

def convert_programmes(df):
    for _, row in df.iterrows():
        prog_uri = uri("Programme", row['programmeCode'])
        g.add((prog_uri, RDF.type, EX.Programme))
        g.add((prog_uri, EX.programmeCode, Literal(row['programmeCode'], datatype=XSD.string)))
        g.add((prog_uri, EX.programmeName, Literal(row['programmeName'], datatype=XSD.string)))

        dept = row['Department name']
        dept_uri = uri("Department", dept.replace(" ", "_"))
        g.add((prog_uri, EX.partOfDepartment, dept_uri))

        direct_uri = uri("SeniorTeacher", row['Director'])
        g.add((prog_uri, EX.hasDirector, direct_uri))

def convert_courses(df):
    for _, row in df.iterrows():
        course_uri = uri("Course", row['courseCode'])
        g.add((course_uri, RDF.type, EX.Course))
        g.add((course_uri, EX.courseCode, Literal(row['courseCode'], datatype=XSD.string)))
        g.add((course_uri, EX.courseName, Literal(row['courseName'], datatype=XSD.string)))
        g.add((course_uri, EX.credits, Literal(int(row['credits']), datatype=XSD.integer)))
        g.add((course_uri, EX.level, Literal(row['level'], datatype=XSD.string)))

        div_uri = uri("Division", row['Division'].replace(" ", "_"))
        g.add((course_uri, EX.belongsToDivisionCourse, div_uri))



def convert_students(df):
    for _, row in df.iterrows():
        student_uri = uri("Student", row['studentId'])
        g.add((student_uri, RDF.type, EX.Student))
        g.add((student_uri, EX.studentId, Literal(row['studentId'], datatype=XSD.string)))
        g.add((student_uri, EX.studentName, Literal(row['studentName'], datatype=XSD.string)))
        g.add((student_uri, EX.year, Literal(row['year'], datatype=XSD.string)))
        g.add((student_uri, EX.graduated, Literal(bool(row['graduated']), datatype=XSD.boolean)))

        prog_uri = uri("Programme", row['Programme'])
        g.add((student_uri, EX.studiesIn, prog_uri))

def convert_senior_teachers(df):
    for _, row in df.iterrows():
        teacher_uri = uri("SeniorTeacher", row['teacherId'])
        g.add((teacher_uri, RDF.type, EX.SeniorTeacher))
        g.add((teacher_uri, RDF.type, EX.Teacher))

        g.add((teacher_uri, EX.teacherId, Literal(row['teacherId'], datatype=XSD.string)))
        g.add((teacher_uri, EX.teacherName, Literal(row['teacherName'], datatype=XSD.string)))
        # 关联 Division
        div_uri = uri("Division", row['Division name'].replace(" ", "_"))
        g.add((teacher_uri, EX.belongsToDivisionTeacher, div_uri))

def convert_teaching_assistants(df):
    for _, row in df.iterrows():
        ta_uri = uri("TeachingAssistant", row['Teacher id'])
        g.add((ta_uri, RDF.type, EX.TeachingAssistant))
        g.add((ta_uri, RDF.type, EX.Student))
        g.add((ta_uri, RDF.type, EX.Teacher))

        g.add((ta_uri, EX.teacherId, Literal(row['Teacher id'], datatype=XSD.string)))
        g.add((ta_uri, EX.teacherName, Literal(row['Teacher name'], datatype=XSD.string)))
        # 关联 Division
        div_uri = uri("Division", row['Division name'].replace(" ", "_"))
        g.add((ta_uri, EX.belongsToDivisionTeacher, div_uri))


def convert_programme_courses(df):
    for _, row in df.iterrows():
        pc_uri = uri("ProgrammeCourse", f"{row['programmeCode']}_{row['courseCode']}_{row['academicYear']}")
        g.add((pc_uri, RDF.type, EX.ProgrammeCourse))

        g.add((pc_uri, EX.academicYear, Literal(row['academicYear'], datatype=XSD.string)))
        g.add((pc_uri, EX.courseType, Literal(row['courseType'], datatype=XSD.string)))
        g.add((pc_uri, EX.studyYear, Literal(row['studyYear'], datatype=XSD.string)))

        g.add((pc_uri, EX.isCourseOf, uri("Course", row['courseCode'])))

        p_uri = uri("Programme", row['programmeCode'])
        g.add((pc_uri, EX.belongsToProgramme, p_uri))
        g.add((p_uri, EX.hasProgrammeCourse, pc_uri))

def convert_registrations(df):
    for _, row in df.iterrows():
        reg_uri = uri("Registration", f"{row['studentId']}_{row['instanceId']}")
        g.add((reg_uri, RDF.type, EX.Registration))

        g.add((reg_uri, EX.grade, Literal(row['grade'], datatype=XSD.string)))
        g.add((reg_uri, EX.status, Literal(row['status'], datatype=XSD.string)))

        g.add((uri("Student", row['studentId']), EX.registersFor, reg_uri))

        ci_uri = uri("CourseInstance", row['instanceId'])
        g.add((reg_uri, EX.forCourseInstance, ci_uri))
        g.add((ci_uri, EX.hasRegistration, reg_uri))

def convert_course_instances(df, plan_df):
    merged_df = pd.merge(df, plan_df, left_on='instanceId', right_on='course', how='left')


    for _, row in merged_df.iterrows():
        ci_uri = uri("CourseInstance", row['instanceId'])
        g.add((ci_uri, RDF.type, EX.CourseInstance))

        g.add((ci_uri, EX.instanceId, Literal(row['instanceId'], datatype=XSD.string)))
        g.add((ci_uri, EX.academicYear, Literal(row['academicYear'], datatype=XSD.string)))
        g.add((ci_uri, EX.studyPeriod, Literal(int(row['studyPeriod']), datatype=XSD.integer)))
        g.add((ci_uri, EX.planningNumStudents, Literal(int(row['numberOfStudents']), datatype=XSD.integer)))
        g.add((ci_uri, EX.seniorHours, Literal(int(row['seniorHours']), datatype=XSD.integer)))
        g.add((ci_uri, EX.assistantHours, Literal(int(row['assistantHours']), datatype=XSD.integer)))

        g.add((ci_uri, EX.examinedBy, uri("SeniorTeacher", row['Examiner'])))

        course_uri = uri("Course", row['Course code'])
        g.add((course_uri, EX.hasInstance, ci_uri))


def convert_assigned_hours(df):
    for _, row in df.iterrows():
        ah_uri = uri("AssignedHours", row['assignedId'])
        g.add((ah_uri, RDF.type, EX.AssignedHours))

        g.add((ah_uri, EX.academicYear, Literal(row['academicYear'], datatype=XSD.string)))
        g.add((ah_uri, EX.hours, Literal(int(row['hours']), datatype=XSD.integer)))

        g.add((ah_uri, EX.assignedTo, uri("Teacher", row['teacherId'])))
        g.add((ah_uri, EX.forCourseInstance, uri("CourseInstance", row['instanceId'])))

def convert_reported_hours(df):
    for _, row in df.iterrows():
        rh_uri = uri("ReportedHours", row['reportedId'])
        g.add((rh_uri, RDF.type, EX.ReportedHours))

        g.add((rh_uri, EX.hours, Literal(int(row['hours']), datatype=XSD.integer)))

        g.add((rh_uri, EX.reportedBy, uri("Teacher", row['teacherId'])))
        g.add((rh_uri, EX.forCourseInstance, uri("CourseInstance", row['instanceId'])))

# ======================
# 主处理流程
# ======================
data_dir = "./data"

programmes_df = pd.read_csv(f"{data_dir}/Programmes.csv").rename(columns={"Programme code": "programmeCode", "Programme name": "programmeName"})
courses_df = pd.read_csv(f"{data_dir}/Courses.csv").rename(columns={"Course code": "courseCode", "Course name": "courseName", "Credits": "credits", "Level": "level"})
students_df = pd.read_csv(f"{data_dir}/Students.csv").rename(columns={"Student id": "studentId", "Student name": "studentName", "Year": "year", "Graduated": "graduated"})
senior_teachers_df = pd.read_csv(f"{data_dir}/Senior_Teachers.csv").rename(columns={"Teacher id": "teacherId", "Teacher name": "teacherName"})
ta_df = pd.read_csv(f"{data_dir}/Teaching_Assistants.csv")
programme_courses_df = pd.read_csv(f"{data_dir}/Programme_Courses.csv").rename(columns={"Programme code": "programmeCode", "Course": "courseCode", "Academic Year": "academicYear", "Course Type": "courseType", "Study Year": "studyYear"})

registrations_df = pd.read_csv(f"{data_dir}/Registrations.csv").rename(columns={"Student id": "studentId", "Course Instance": "instanceId", "Grade": "grade", "Status": "status"})
# registrations_df["registrationId"] = registrations_df.index + 1

instances_df = pd.read_csv(f"{data_dir}/Course_Instances.csv").rename(columns={"Instance_id": "instanceId", "Academic year": "academicYear", "Study period": "studyPeriod"})
courses_planning_df = pd.read_csv(f"{data_dir}/Course_plannings.csv").rename(columns={"Course": "course", "Planned number of Students": "numberOfStudents", "Senior Hours": "seniorHours", "Assistant Hours": "assistantHours"})

assigned_df = pd.read_csv(f"{data_dir}/Assigned_Hours.csv").rename(columns={"Academic Year": "academicYear", "Teacher Id": "teacherId", "Course Instance": "instanceId", "Hours": "hours"})
assigned_df["assignedId"] = assigned_df.index + 1
reported_df = pd.read_csv(f"{data_dir}/Reported_Hours.csv").rename(columns={"Teacher Id": "teacherId", "Course code": "instanceId", "Hours": "hours"})
reported_df["reportedId"] = reported_df.index + 1

# 执行转换
convert_departments_and_divisions(courses_df, senior_teachers_df, ta_df, programmes_df)
convert_programmes(programmes_df)
convert_courses(courses_df)
convert_students(students_df)
convert_senior_teachers(senior_teachers_df)
convert_teaching_assistants(ta_df)
convert_programme_courses(programme_courses_df)
convert_registrations(registrations_df)
convert_course_instances(instances_df, courses_planning_df)
convert_assigned_hours(assigned_df)
convert_reported_hours(reported_df)

# 导出 TTL 文件
g.serialize(destination="full_data.ttl", format="turtle")
print("RDF data: full_data.ttl")

