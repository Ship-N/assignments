import pandas as pd
from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef
from rdflib.namespace import XSD
from rdflib.namespace import OWL

def convert_course_instances(df, plan_df):
    merged_df = pd.merge(df, plan_df, left_on='instanceId', right_on='course', how='left')
    for _, row in merged_df.iterrows():
        print(row)

instances_df = pd.read_csv(f"data/Course_Instances.csv").rename(columns={"Instance_id": "instanceId", "Academic year": "academicYear", "Study period": "studyPeriod"})
courses_planning_df = pd.read_csv(f"data/Course_plannings.csv").rename(columns={"Course": "course", "Planned number of Students": "numberOfStudents", "Senior Hours": "seniorHours", "Assistant Hours": "assistantHours"})

convert_course_instances(instances_df, courses_planning_df)