# RAILWAY INFRASTRUCTURE INSPECTION
<div align="center">

![img](/assets/problem_statement.png)
## Our Approach:

## Our Solution Addresses Three Key Problems:
<div align="center">
    <table>
        <tr>
            <td align="center" width="33%">
                <h3>Railway Track Defects</h3>
                <kbd><img src="/assets/switch.png" alt="Track Inspection" width="100"></kbd>
                <p>Identify and analyze defects in railway tracks, reducing risks of derailments and ensuring smooth and safe train movement.</p>
            </td>
            <td align="center" width="33%">
                <h3>Bridge Inspection</h3>
                <kbd><img src="/assets/bridge.png" alt="Bridge Inspection" width="100"></kbd>
                <p>Detect and monitor structural defects in railway bridges, preventing potential failures and ensuring safe train operations.</p>
            </td>
            <td align="center" width="33%">
                <h3>Real-Time Obstacle Detection</h3>
                <kbd><img src="/assets/obstacle.png" alt="Obstacle Detection" width="100"></kbd>
                <p>Detect and alert the loco pilot about track obstacles, preventing potential hazards and damage to locomotives.</p>
            </td>
        </tr>
    </table>
</div>

## Bridge Defect Detection

<div align="center">
    <img src="/assets/blockdiag.png" alt="Block Diagram" width="700">
    <p><i>Comprehensive System Block Diagram</i></p>
</div>

### Why Bridge Inspection Matters

<div align="center">
    <img src="/assets/railway-infra.png" alt="Railway Infrastructure Incidents" width="700">
    <p><i>Real-world incidents highlighting the critical need for bridge inspection</i></p>
</div>

### Dataset Details

<div align="center">
    <img src="/assets/dataset_bridge.png" alt="Bridge Dataset Details" width="700">
</div>

### Our Technical Approach for Bridge Inspection

<div align="center">
<div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 20px; margin-bottom: 30px;">
<div style="flex: 1; min-width: 300px; border-radius: 8px; padding: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); background: #f8f9fa;">
<h4>🔍Detection Models</h4>
<div style="margin: 15px 0;">
<img src="/assets/8.png" alt="Model Results" width="100%" style="border-radius: 6px;">
</div>
<p><strong>Core Technology:</strong> Transfer learning implementation using DINO v2 architecture</p>
<p><strong>Specialization:</strong> Optimized for detecting fine cracks and structural anomalies</p>
</div>

<div style="flex: 1; min-width: 300px; border-radius: 8px; padding: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); background: #f8f9fa;">
<h4>⚡ YOLOv11 Implementation</h4>
<div style="margin: 15px 0;">
<img src="/assets/9.png" alt="YOLOv11 Results" width="100%" style="border-radius: 6px;">
</div>
<p><strong>Core Technology:</strong> State-of-the-art object detection for structural defects</p>
<p><strong>Specialization:</strong> Real-time detection with high precision in varying lighting conditions</p>
</div>
</div>

<div style="width: 100%; height: 4px; background: linear-gradient(90deg, #3498db, #2ecc71); margin: 20px 0; border-radius: 2px;"></div>

<p><i>Our dual-model approach combines the strengths of both architectures to achieve superior bridge defect detection performance</i></p>
</div>

## Railway Track Defects Detection

<div align="center">
    <p><i>Ensuring railway safety through advanced track defect detection</i></p>
</div>

### Impact of Railway Track Defects

<div align="center">
    <table>
        <tr>
            <td align="center" width="50%">
                <h4>Financial Implications</h4>
                <kbd><img src="/assets/money-management.png" alt="Financial Impact" width="100"></kbd>
                <p>Track defects contribute to frequent derailments and maintenance costs, leading to <b>₹300 crore</b> in annual losses for Indian Railways.</p>
            </td>
            <td align="center" width="50%">
                <h4>Safety Concerns</h4>
                <kbd><img src="/assets/shield.png" alt="Safety Icon" width="100"></kbd>
                <p>Between 2018-2021, <b>70%</b> of train accidents were directly linked to track failures, including the Amritsar disaster (2018) with 61 casualties.</p>
            </td>
        </tr>
    </table>
</div>

### Common Causes of Track Defects

<div align="center">
    <table>
        <tr>
            <td align="center" width="33%">
                <h4>Wear & Tear</h4>
                <kbd><img src="/assets/breaking.png" alt="Wear and Tear" width="80"></kbd>
                <p>Continuous train movement causes cracks, misalignments, and rail fractures over time.</p>
            </td>
            <td align="center" width="33%">
                <h4>Weather Conditions</h4>
                <kbd><img src="/assets/weather-news.png" alt="Weather Conditions" width="80"></kbd>
                <p>Extreme heat, cold, and waterlogging significantly weaken track infrastructure.</p>
            </td>
            <td align="center" width="33%">
                <h4>Poor Maintenance</h4>
                <kbd><img src="/assets/maintainance.png" alt="Maintenance Issues" width="80"></kbd>
                <p>Delayed inspections lead to unnoticed defects, substantially increasing safety risks.</p>
            </td>
        </tr>
    </table>
</div>

### Infrastructure & Rolling Stock Damage

<div align="center">
    <table>
        <tr>
            <td align="center" width="33%">
                <h4>Track Failures</h4>
                <p>Broken or misaligned rails cause derailments and slow operations.</p>
            </td>
            <td align="center" width="33%">
                <h4>Engine & Coach Damage</h4>
                <p>Sudden jolts from faulty tracks damage valuable rolling stock.</p>
            </td>
            <td align="center" width="33%">
                <h4>Signaling Disruptions</h4>
                <p>Track failures interfere with electronic safety systems.</p>
            </td>
        </tr>
    </table>
</div>

### Accident Analysis

<div align="center">
    <img src="/assets/11.png" alt="Accidents and their reasons" width="700">
    <p><i>Distribution of railway accidents by cause (2018-2022)</i></p>
</div>

### Dataset Information

<div align="center">
    <img src="/assets/12.png" alt="Dataset Information" width="700">
    <p><i>Comprehensive dataset used for training our detection models</i></p>
</div>

## Technical Implementation & Results

### Advanced Detection Models

<div align="center">
        <table>
            <tr>
                <td align="center" width="33%">
                    <h4>R-CNN Framework</h4>
                    <kbd><img src="/assets/rcnn.png" alt="R-CNN" width="80"></kbd>
                    <p>Region-based CNN implementation with feature extraction capabilities for precise track defect localization.</p>
                </td>
                <td align="center" width="33%">
                    <h4>YOLOv11 Architecture</h4>
                    <kbd><img src="/assets/roboflow.png" alt="YOLOv11" width="80"></kbd>
                    <p>Real-time detection with improved accuracy and reduced false positives for critical track components.</p>
                </td>
                <td align="center" width="33%">
                    <h4>Florence Vision Model</h4>
                    <kbd><img src="/assets/microsoft.png" alt="Florence" width="80"></kbd>
                    <p>Multi-modal foundation model delivering superior performance in varied lighting and weather conditions.</p>
                </td>
            </tr>
        </table>
    <img src="/assets/13.png" alt="Models Architecture" width="700">
</div>

<div align="center">
    <img src="/assets/15.png" alt="Detection Results" width="700">

## Obstacle Detection and Inspection


### Dataset Information

<div align="center">
    <img src="/assets/17.png" alt="Obstacle Dataset Details" width="700">
    <p><i>Comprehensive dataset used for training our obstacle detection models</i></p>
</div>

### Detection Results

<div align="center">
    <img src="/assets/18.png" alt="Obstacle Detection Results" width="700">
    <p><i>Performance metrics of our obstacle detection system</i></p>
</div>

### Technical Approach

<div align="center">
    <img src="/assets/19.png" alt="Obstacle Detection Approach Part 1" width="700">
    <p><i>Implementation architecture and methodology</i></p>

</div>

## 🏆 Project Recognition

This project was a contribution to the **Wabtec Corporation's Exceed 3.0 Hackathon**, where our team achieved **2nd Runner Up** position out of numerous competing teams.

### 🎉 Achievements & Press Coverage

![img](/assets/All-3-winners-1536x1025.jpg)

We're honored that our work was recognized by both Wabtec and industry publications:

- **[Wabtec Corporation Press Release](https://www.wabteccorp.com/newsroom/press-releases/wabtec-announces-winners-for-its-exceed-30-campus-challenge-in-india)**  
  Official announcement from Wabtec about the hackathon winners

- **[Rail Analysis](https://railanalysis.in/rail-news/wabtec-announces-winners-of-exceed-3-0-campus-challenge-in-india-showcasing-future-ready-rail-innovations/)**  
  Industry coverage highlighting the future-ready rail innovations

- **[Market Screener](https://www.marketscreener.com/quote/stock/WESTINGHOUSE-AIR-BRAKE-TE-14842/news/Wabtec-Announces-Winners-for-Its-Exceed-3-0-Campus-Challenge-in-India-48952822/)**  
  Financial markets coverage of the event

- **[APN News](https://www.apnnews.com/wabtec-announces-winners-for-its-exceed-3-0-campus-challenge-in-india/)**  
  Additional press coverage of the competition results

> *"The Exceed 3.0 challenge showcased innovative solutions that demonstrate the bright future of rail technology."*  
> — Wabtec Corporation