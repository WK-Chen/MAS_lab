from data_gen import mixed_scenarios
import xml.etree.ElementTree as ET

xml_data = """
<utility_space type="any" number_of_issues="0">
<objective index="1" etype="objective" type="objective" description="" name="any">
<issue index="1" etype="discrete" type="discrete" vtype="discrete" name="i1">
    <item index="1" value="v1" evaluation="1.0" />
    <item index="2" value="v2" evaluation="0.0" />
</issue>
<issue index="2" etype="discrete" type="discrete" vtype="discrete" name="i2">
    <item index="1" value="v1" evaluation="0.7222412055200594" />
    <item index="2" value="v2" evaluation="0.5758198581567425" />
    <item index="3" value="v3" evaluation="0.0" />
    <item index="4" value="v4" evaluation="0.2739365695690635" />
    <item index="5" value="v5" evaluation="1.0" />
</issue>
<issue index="3" etype="discrete" type="discrete" vtype="discrete" name="i3">
    <item index="1" value="v1" evaluation="0.0" />
    <item index="2" value="v2" evaluation="0.6053732598589376" />
    <item index="3" value="v3" evaluation="0.7464549278169106" />
    <item index="4" value="v4" evaluation="1.0" />
</issue>
<issue index="4" etype="discrete" type="discrete" vtype="discrete" name="i4">
    <item index="1" value="v1" evaluation="0.0" />
    <item index="2" value="v2" evaluation="1.0" />
    <item index="3" value="v3" evaluation="0.0" />
    <item index="4" value="v4" evaluation="0.5" />
</issue>
<weight index="1" value="0.01621590153731152">
</weight>
<weight index="2" value="0.5694072472562208">
</weight>
<weight index="3" value="0.016764235021048637">
</weight>
<weight index="4" value="0.397612616185419">
</weight>
</objective>
<reservation value="0.4454218476422879" />
</utility_space>
"""

root = ET.fromstring(xml_data)


# 提取权重
weights = {int(weight.get('index')): float(weight.get('value')) for weight in root.findall('.//weight')}

# 对每个议题进行遍历
for issue in root.findall('.//issue'):
    issue_index = int(issue.get('index'))
    weight = weights.get(issue_index, 1)  # 默认权重为 1，如果找不到对应的权重

    print(f"Issue {issue_index} weight: {weight}")

    for item in issue.findall('item'):
        item_index = int(item.get('index'))
        item_value = item.get('value')
        evaluation = float(item.get('evaluation'))

        # 计算加权评估值
        weighted_evaluation = weight * evaluation

        print(f"    Value {item_value}: Evaluation {evaluation}, Weighted Evaluation {weighted_evaluation}")
