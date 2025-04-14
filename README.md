# 🚆 Railway Infrastructure Inspection System

<div align="center">

![Railway Infrastructure Inspection Banner](/assets/problem_statement.png)

[![Railway Safety](https://img.shields.io/badge/Focus-Railway%20Safety-blue)](https://github.com/shiva/RAILWAY-INFRASTRUCTURE-INSPECTION)
[![AI Powered](https://img.shields.io/badge/Technology-AI%20Powered-green)](https://github.com/shiva/RAILWAY-INFRASTRUCTURE-INSPECTION)
[![Deep Learning](https://img.shields.io/badge/AI-Deep%20Learning-orange)](https://github.com/shiva/RAILWAY-INFRASTRUCTURE-INSPECTION)
[![Computer Vision](https://img.shields.io/badge/Tech-Computer%20Vision-purple)](https://github.com/shiva/RAILWAY-INFRASTRUCTURE-INSPECTION)

</div>

## 📋 Overview

An advanced AI-powered system designed to enhance railway safety through automated inspection of critical infrastructure components. Our solution addresses three major areas of concern in railway operations:

<div align="center">
    <table>
        <tr>
            <td align="center" width="33%">
                <h3>🛤️ Track Defect Detection</h3>
                <kbd><img src="/assets/switch.png" alt="Track Inspection" width="120"></kbd>
                <p>Identifies cracks, misalignments, and structural defects in railway tracks, reducing derailment risks by up to 70%.</p>
            </td>
            <td align="center" width="33%">
                <h3>🌉 Bridge Inspection</h3>
                <kbd><img src="/assets/bridge.png" alt="Bridge Inspection" width="120"></kbd>
                <p>Detects and monitors structural weaknesses in railway bridges, preventing potential catastrophic failures.</p>
            </td>
            <td align="center" width="33%">
                <h3>⚠️ Obstacle Detection</h3>
                <kbd><img src="/assets/obstacle.png" alt="Obstacle Detection" width="120"></kbd>
                <p>Real-time alert system for track obstacles, providing crucial warnings to locomotive pilots to prevent accidents.</p>
            </td>
        </tr>
    </table>
</div>

## 🔍 Why This Matters

<div align="center">
    <img src="/assets/railway-infra.png" alt="Railway Infrastructure Issues" width="700">
</div>

### Critical Safety Concerns:
- **70% of train accidents** (2018-2021) were linked to track failures
- **₹300 crore annual losses** for Indian Railways due to track defects
- **Amritsar Train Disaster (2018)**: Poor track conditions contributed to 61 deaths
- Continuous train movement causes cracks, misalignments, and rail fractures
- Extreme weather and poor maintenance exacerbate infrastructure deterioration

<div align="center">
    <img src="/assets/11.png" alt="Accident Statistics" width="700">
</div>

## 🧠 Our Technical Approach

<div align="center">
    <img src="/assets/blockdiag.png" alt="System Architecture" width="800">
</div>

### Bridge Defect Detection (HEDGE-Z)

Our system utilizes transfer learning with DINO V2 and YOLOv11 models to identify structural issues in bridges:

<div align="center">
    <img src="/assets/8.png" alt="Bridge Detection Model" width="400">
    <img src="/assets/9.png" alt="Bridge Detection Results" width="400">
</div>

#### Dataset Details:
<div align="center">
    <img src="/assets/dataset_bridge.png" alt="Bridge Dataset" width="700">
</div>

### Railway Track Defect Detection

We employ R-CNN, YOLOv11, and Florence models to detect various track defects including:
- Broken rails and rail joints
- Track geometry issues
- Fastening system defects
- Sleeper deterioration

<div align="center">
    <img src="/assets/13.png" alt="Track Detection Models" width="700">
</div>

#### Impact of Track Defects:
1. **Safety Risks**: Potential derailments endangering lives
2. **Financial Losses**: Maintenance costs and operational delays
3. **Infrastructure Damage**: Accelerated deterioration of related components
4. **Operational Inefficiency**: Speed restrictions and service disruptions

#### Dataset Information:
<div align="center">
    <img src="/assets/12.png" alt="Track Dataset" width="700">
</div>

#### Results:
<div align="center">
    <img src="/assets/15.png" alt="Track Detection Results" width="700">
</div>

### Obstacle Detection System

A real-time vision system that identifies foreign objects on tracks that could cause accidents:

<div align="center">
    <img src="/assets/16.png" alt="Obstacle Detection System" width="700">
</div>

#### Implementation Approach:
<div align="center">
    <img src="/assets/19.png" alt="Obstacle Detection Implementation" width="400">
    <img src="/assets/20.png" alt="Obstacle Detection Flow" width="400">
</div>

#### Dataset Details:
<div align="center">
    <img src="/assets/17.png" alt="Obstacle Dataset" width="700">
</div>

#### Performance Results:
<div align="center">
    <img src="/assets/18.png" alt="Obstacle Detection Results" width="700">
</div>

## 🔑 Key Features

- **Automated Inspection**: Reduces human error and inspection time
- **Early Warning System**: Identifies defects before they cause accidents
- **AI-Powered Analysis**: Leverages state-of-the-art computer vision models
- **Comprehensive Coverage**: Tracks, bridges, and surrounding areas
- **Actionable Insights**: Prioritized maintenance recommendations based on defect severity

## 📊 Impact

- **Enhanced Safety**: Significant reduction in accident risk through proactive defect detection
- **Cost Efficiency**: Minimized maintenance costs through timely interventions
- **Improved Operations**: Reduced downtime and service disruptions
- **Data-Driven Decisions**: Better resource allocation for maintenance prioritization

## 🔗 Contact

For more information about this project or to contribute, please reach out to the repository owner.

---

<div align="center">
    <p><i>Building safer railways through intelligent infrastructure inspection</i></p>
</div>