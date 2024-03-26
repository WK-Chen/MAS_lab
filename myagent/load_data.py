import xml.etree.ElementTree as ET
import itertools

import numpy as np

def read_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()

    # Initialize dictionary to store the XML data
    xml_dict = {}

    # Extract data from issues
    issues = []
    for issue in root.findall('.//issue'):
        issue_data = [
            float(item.get("evaluation")) for item in issue.findall('item')
        ]
        issues.append(issue_data)

    # Extract data from weights
    weights = [float(weight.get("value")) for weight in root.findall('.//weight')]

    # Extract data from reservation
    reservation = float(root.find('.//reservation').get("value"))

    return issues, weights, reservation

def calculate_ufun(issues, weights):
    issues_with_weights = []
    for issue, weight in zip(issues, weights):
        i_w = [i * weight for i in issue]
        issues_with_weights.append(i_w)

    res = itertools.product(*issues_with_weights, repeat=1)

    ufun_points = sorted(list(set([sum(r) for r in list(res)])))

    return ufun_points

if __name__ == '__main__':
    path = '../data/Domain1/prof1.xml'
    issues, weights, reservation = read_xml(path)
    ufun_points = calculate_ufun(issues, weights)
    print({"ufun_points": ufun_points, "reservation": reservation})

