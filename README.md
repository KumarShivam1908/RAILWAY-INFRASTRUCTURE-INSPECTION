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

### Our Technical Approach

<div align="center">
    <table>
        <tr>
            <td align="center" width="50%">
                <h4>HEDGE-Z Detection Model</h4>
                <p>Transfer learning implementation using DINO v2 architecture</p>
                <kbd><img src="/assets/8.png" alt="HEDGE-Z Model Results" width="300"></kbd>
            </td>
            <td align="center" width="50%">
                <h4>YOLOv11 Implementation</h4>
                <p>State-of-the-art object detection for structural defects</p>
                <kbd><img src="/assets/9.png" alt="YOLOv11 Results" width="300"></kbd>
            </td>
        </tr>
    </table>
</div>

## Railway Track Defects Detection

<div align="center">
    <img src="/assets/11.png" alt="Track Accidents and Reasons" width="700">
    <p><i>Analysis of Railway Accidents: Track Defects as Primary Cause</i></p>
</div>

### Impact of Railway Track Defects

<div align="center">
    <table>
        <tr>
            <td align="center" width="50%">
                <h4>Financial Implications</h4>
                <p>Track defects contribute to frequent derailments and maintenance costs, leading to <b>₹300 crore in losses</b> for Indian Railways annually.</p>
            </td>
            <td align="center" width="50%">
                <h4>Safety Concerns</h4>
                <p>Between 2018-2021, <b>70% of train accidents</b> were linked to track failures, including the Amritsar disaster (2018) where poor track conditions contributed to 61 fatalities.</p>
            </td>
        </tr>
    </table>
</div>

### Track Defect Detection Workflow

<div align="center">
    ```mermaid
    graph TD
        A[Data Acquisition] --> B[Preprocessing]
        B --> C[Feature Extraction]
        C --> D[Defect Detection Models]
        D --> E[Analysis & Classification]
        E --> F[Alert Generation]
        F --> G[Maintenance Recommendation]
        
        style A fill:#f9d5e5,stroke:#333,stroke-width:2px
        style D fill:#eeeeee,stroke:#333,stroke-width:3px
        style G fill:#d5f9e5,stroke:#333,stroke-width:2px
    ```
    <p><i>End-to-end workflow for track defect detection and maintenance</i></p>
</div>

### Primary Causes of Track Defects

<div align="center">
    <table>
        <tr>
            <td align="center" width="33%">
                <h4>Wear & Tear</h4>
                <kbd><img src="/assets/wear_tear.png" alt="Wear and Tear" width="100" onerror="this.onerror=null; this.src='/assets/switch.png'"></kbd>
                <p>Continuous train movement causes cracks, misalignments, and rail fractures over time.</p>
            </td>
            <td align="center" width="33%">
                <h4>Weather Conditions</h4>
                <kbd><img src="/assets/weather.png" alt="Weather Conditions" width="100" onerror="this.onerror=null; this.src='/assets/switch.png'"></kbd>
                <p>Extreme heat, cold, and waterlogging significantly weaken track infrastructure.</p>
            </td>
            <td align="center" width="33%">
                <h4>Poor Maintenance</h4>
                <kbd><img src="/assets/maintenance.png" alt="Poor Maintenance" width="100" onerror="this.onerror=null; this.src='/assets/switch.png'"></kbd>
                <p>Delayed inspections lead to unnoticed defects, dramatically increasing operational risks.</p>
            </td>
        </tr>
    </table>
</div>

### Defect Classification System

<div align="center">
    ```mermaid
    flowchart LR
        A[Image Input] --> B{Detection System}
        B -->|Class 1| C[Rail Cracks]
        B -->|Class 2| D[Misalignments]
        B -->|Class 3| E[Missing Fasteners]
        B -->|Class 4| F[Rail Corrugation]
        B -->|Class 5| G[Other Defects]
        
        C & D & E & F & G --> H[Severity Assessment]
        H --> I[Maintenance Planning]
    ```
    <p><i>Classification system for different types of track defects</i></p>
</div>

### Infrastructure & Train Damage

<div align="center">
    <table>
        <tr>
            <td align="center" width="33%">
                <h4>Track Failures</h4>
                <p>Broken or misaligned rails cause derailments and slow operations.</p>
            </td>
            <td align="center" width="33%">
                <h4>Engine & Coach Damage</h4>
                <p>Sudden jolts from faulty tracks damage expensive rolling stock.</p>
            </td>
            <td align="center" width="33%">
                <h4>Signaling Disruptions</h4>
                <p>Track failures interfere with electronic safety systems, compromising safety protocols.</p>
            </td>
        </tr>
    </table>
</div>

### Dataset Information & Technical Implementation

<div align="center">
    <img src="/assets/12.png" alt="Track Defect Dataset" width="700">
    <p><i>Comprehensive Dataset for Track Defect Detection</i></p>
</div>

### Detection Architecture

<div align="center">
    ```mermaid
    graph TB
        subgraph "Track Defect Detection System"
        A[Input Images] --> B[Image Preprocessing]
        B --> C[Feature Extraction]
        C --> D{Model Selection}
        D --> E[R-CNN]
        D --> F[YOLOv11]
        D --> G[Florence]
        E & F & G --> H[Ensemble Results]
        H --> I[Defect Classification]
        I --> J[Report Generation]
        end
    ```
    <p><i>Detailed architecture of our multi-model approach to track defect detection</i></p>
</div>

### Detection Models & Performance

<div align="center">
    <img src="/assets/13.png" alt="Detection Models" width="700">
    <p><i>Advanced models implemented: R-CNN, YOLOv11, and Florence architectures</i></p>
</div>

### Detection Results

<div align="center">
    <img src="/assets/15.png" alt="Track Defect Detection Results" width="700">
    <p><i>Track defect detection performance metrics and sample visualizations</i></p>
    
    <table>
        <tr>
            <th>Model</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-Score</th>
            <th>Inference Time</th>
        </tr>
        <tr>
            <td>R-CNN</td>
            <td>87.3%</td>
            <td>85.6%</td>
            <td>86.4%</td>
            <td>210ms</td>
        </tr>
        <tr>
            <td>YOLOv11</td>
            <td>92.4%</td>
            <td>91.8%</td>
            <td>92.1%</td>
            <td>48ms</td>
        </tr>
        <tr>
            <td>Florence</td>
            <td>94.1%</td>
            <td>93.7%</td>
            <td>93.9%</td>
            <td>85ms</td>
        </tr>
        <tr>
            <td><b>Ensemble Model</b></td>
            <td><b>95.8%</b></td>
            <td><b>94.6%</b></td>
            <td><b>95.2%</b></td>
            <td>110ms</td>
        </tr>
    </table>
</div>


  Obstacle Detection and Inspection

![img](/assets/16.png)

Dataset info

![img](/assets/17.png)

Result

![img](/assets/18.png)

Approach

![img](/assets/19.png)
![img](/assets/20)