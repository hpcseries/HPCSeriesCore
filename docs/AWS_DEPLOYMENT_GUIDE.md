# AWS Deployment Guide
## HPCSeries Core v0.7 - CPU-Optimized Production Deployment

**Complete workflow**: Local development → GitHub → AWS EC2 CPU deployment

---

## 🎯 Overview

This guide covers **production deployment** of HPCSeries Core v0.7 on AWS EC2 CPU-optimized instances:

1. **Bare Metal Installation** (Primary) - Direct installation for maximum performance
2. **Docker Deployment** (Alternative) - Containerized deployment
3. **Performance Optimization** - Calibration and tuning for AWS CPUs
4. **Production Best Practices** - Monitoring, scaling, cost optimization

---

## 📋 AWS Instance Selection

### Recommended Instance Families

HPCSeries Core automatically detects CPU architecture and applies optimal compilation flags. Choose instance families based on your workload needs:

| Family | Architecture | SIMD | vCPU Range | Use Case | Price Tier |
|--------|-------------|------|------------|----------|------------|
| **c7i** | Intel Sapphire Rapids | AVX-512 | 2-192 | Best performance, latest Intel | $$$ |
| **c6i** | Intel Ice Lake | AVX-512 | 2-128 | Good value, AVX-512 support | $$ |
| **c7g** | ARM Graviton3 | NEON | 1-64 | Cost-effective ARM, good performance | $ |
| **c6a** | AMD EPYC 3rd Gen | AVX2 | 2-192 | AMD alternative, AVX2 | $$ |
| **c5** | Intel Cascade Lake | AVX2 | 2-96 | Legacy, budget option | $ |

**Typical instance sizes** within each family:
- **Development/Testing**: .xlarge (4 vCPU, 8 GB) or .2xlarge (8 vCPU, 16 GB)
- **Small Production**: .2xlarge to .4xlarge (8-16 vCPU, 16-32 GB)
- **Large Production**: .8xlarge to .16xlarge (32-64 vCPU, 64-128 GB)
- **NUMA Workloads**: .metal instances (2-socket, 48-96 vCPU)

### Decision Matrix

**Choose c7i family if:**
- ✅ Need maximum performance (latest Intel, AVX-512)
- ✅ Running large production workloads
- ✅ Budget allows for premium instances

**Choose c6i family if:**
- ✅ Want AVX-512 performance at lower cost
- ✅ Good balance of performance and price
- ✅ Production workloads with moderate throughput

**Choose c7g family if:**
- ✅ Cost optimization is top priority
- ✅ Willing to use ARM architecture
- ✅ Workload scales well without x86-specific optimizations

**Choose c6a family if:**
- ✅ Prefer AMD architecture
- ✅ AVX2 is sufficient for your workload
- ✅ Good price-performance ratio

**Choose c5 family if:**
- ✅ Budget-constrained development/testing
- ✅ AVX2 is sufficient
- ✅ Not requiring latest CPU features

### Architecture Auto-Detection

HPCSeries automatically detects your instance's CPU architecture during build:
- **c7i/c6i** → Compiles with AVX-512 flags (`-march=native`)
- **c7g** → Compiles with ARM NEON flags (`-mcpu=native`)
- **c6a/c5** → Compiles with AVX2 flags (`-march=native`)

---

## 🚀 Part 1: Bare Metal Installation (Primary Method)

### Step 1: Launch EC2 Instance

```bash
# Using AWS CLI (or use AWS Console)
# Adjust instance-type based on your chosen family and size
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type c6i.2xlarge \
  --key-name YOUR_KEY_NAME \
  --security-group-ids sg-XXXXXXXXX \
  --subnet-id subnet-XXXXXXXXX \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=hpcs-production}]'

# Note the Instance ID and Public IP
```

**Instance type examples**: `c7i.2xlarge`, `c6i.4xlarge`, `c7g.xlarge`, `c6a.2xlarge`

**Recommended AMI**: Amazon Linux 2023 or Ubuntu 22.04 LTS

### Step 2: SSH to Instance

```bash
# Amazon Linux
ssh -i "YOUR_KEY.pem" ec2-user@YOUR_EC2_PUBLIC_IP

# Ubuntu
ssh -i "YOUR_KEY.pem" ubuntu@YOUR_EC2_PUBLIC_IP
```

### Step 3: Install System Dependencies

**Amazon Linux 2023:**

```bash
# Update system
sudo dnf update -y

# Install Python and development tools
sudo dnf install -y python3.11 python3.11-pip python3.11-devel

# Install compilers (for building from source)
sudo dnf install -y gcc gcc-c++ gcc-gfortran cmake

# Install NumPy dependencies
sudo dnf install -y openblas-devel lapack-devel

# Verify installation
python3.11 --version  # Should show 3.11.x
gcc --version         # Should show 11.x or higher
gfortran --version    # Should show 11.x or higher
```

**Ubuntu 22.04:**

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python and development tools
sudo apt-get install -y python3.11 python3-pip python3.11-dev

# Install compilers
sudo apt-get install -y gcc g++ gfortran cmake

# Install NumPy dependencies
sudo apt-get install -y libopenblas-dev liblapack-dev

# Verify installation
python3.11 --version
gcc --version
gfortran --version
```

### Step 4: Install HPCSeries Core

**Option A: Install from PyPI (Recommended for Production)**

```bash
# Create virtual environment (recommended)
python3.11 -m venv ~/hpcs-env
source ~/hpcs-env/bin/activate

# Install HPCSeries
pip install --upgrade pip
pip install hpcs

# Verify installation
python -c "import hpcs; print(hpcs.__version__)"
python -c "import hpcs; print(hpcs.simd_info())"
```

Expected output:
```
0.7.0
{'isa': 'AVX-512', 'width_bytes': 64, 'width_doubles': 8}  # On c7i/c6i
# or
{'isa': 'AVX2', 'width_bytes': 32, 'width_doubles': 4}     # On c5
```

**Option B: Build from Source (For Custom Optimizations)**

```bash
# Clone repository
cd ~
git clone https://github.com/your-org/HPCSeriesCore.git
cd HPCSeriesCore

# Create virtual environment
python3.11 -m venv ~/hpcs-env
source ~/hpcs-env/bin/activate

# Build C/Fortran library (architecture auto-detection)
mkdir -p build && cd build

# Default SAFE profile (IEEE 754 compliant, recommended)
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Or use FAST profile (5-10% faster, relaxed IEEE 754)
# export HPCS_PROFILE=FAST
# cmake .. -DCMAKE_BUILD_TYPE=Release
# make -j$(nproc)

cd ..

# Install Python package
pip install --upgrade pip
pip install -e .

# Verify installation and architecture detection
python -c "import hpcs; print(hpcs.__version__)"
python -c "import hpcs; print(hpcs.simd_info())"
```

**Architecture Detection**: HPCSeries automatically detects your CPU and applies optimal flags:
- **Intel (c7i, c6i)**: `-march=native` (AVX-512)
- **ARM (c7g)**: `-mcpu=native` (NEON)
- **AMD (c6a)**: `-march=native` (AVX2)

---

### Architecture-Aware Build Profiles

HPCSeries supports two compilation profiles:

| Profile | `-ffast-math` | IEEE 754 | Performance | Use When |
|---------|--------------|----------|-------------|----------|
| **SAFE** (default) | ❌ | ✅ Compliant | Baseline | Production, data with NaN/Inf values |
| **FAST** | ✅ | ⚠️ Relaxed | +5-10% | Performance-critical, clean data |

**To use FAST profile**:
```bash
export HPCS_PROFILE=FAST
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**Recommendation**: Use **SAFE** profile unless you have verified your data contains no NaN/Inf values and have tested numerical accuracy.

---

### Step 5: Performance Calibration

**Critical for production performance!**

```bash
# Activate environment
source ~/hpcs-env/bin/activate

# Run calibration (~30 seconds)
python << 'EOF'
import hpcs

print("Running calibration on AWS instance...")
hpcs.calibrate()
hpcs.save_calibration_config()

# Verify calibration
print("\nCalibration complete!")
print(f"CPU Info: {hpcs.get_cpu_info()}")
print(f"SIMD Info: {hpcs.simd_info()}")
EOF
```

This creates `~/.hpcs/config.json` with optimal settings for your instance type.

### Step 6: Verify Performance

```bash
# Quick performance test
python << 'EOF'
import hpcs
import numpy as np
import time

# Test data
data = np.random.randn(10_000_000)

# Test rolling median (should be 50-100x faster than Pandas)
start = time.perf_counter()
result = hpcs.rolling_median(data, window=100)
elapsed = time.perf_counter() - start

print(f"Rolling median (10M elements, w=100): {elapsed*1000:.2f} ms")
print(f"Expected: ~100-200 ms on c7i/c6i (AVX-512), ~150-250 ms on c5 (AVX2)")

# Test SIMD reductions
start = time.perf_counter()
mean = hpcs.mean(data)
elapsed = time.perf_counter() - start

print(f"Mean (10M elements): {elapsed*1000:.2f} ms")
print(f"Expected: ~5-10 ms on c7i/c6i (AVX-512), ~8-15 ms on c5 (AVX2)")
EOF
```

---

## 🐳 Part 2: Docker Deployment (Alternative Method)

If you prefer containerized deployment:

### Step 1: Install Docker on EC2

**Amazon Linux 2023:**

```bash
sudo dnf install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user

# Log out and back in for group changes
exit
# SSH back in
```

**Ubuntu 22.04:**

```bash
sudo apt-get install -y docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ubuntu

# Log out and back in
exit
# SSH back in
```

### Step 2: Deploy with Docker

```bash
# Clone repository
cd ~
git clone https://github.com/your-org/HPCSeriesCore.git
cd HPCSeriesCore

# Build container
docker compose -f docker-compose.python.yml build

# Run calibration in container
docker compose -f docker-compose.python.yml run --rm hpcs-python python -c "
import hpcs
hpcs.calibrate()
hpcs.save_calibration_config()
print('Calibration complete!')
"

# Run your application
docker compose -f docker-compose.python.yml run --rm hpcs-python python your_script.py
```

---

## ⚡ Performance Optimization

### Configure OpenMP Threads

```bash
# Set to physical core count (not hyperthreaded)
# Example: For .2xlarge instances (8 vCPUs = 4 physical cores):
export OMP_NUM_THREADS=4

# Example: For .4xlarge instances (16 vCPUs = 8 physical cores):
export OMP_NUM_THREADS=8

# Add to ~/.bashrc for persistence
echo "export OMP_NUM_THREADS=$(lscpu | grep '^Core(s) per socket' | awk '{print $4}')" >> ~/.bashrc
```

**How to determine optimal thread count:**

```bash
# Get physical core count (not hyperthreaded)
CORES_PER_SOCKET=$(lscpu | grep "^Core(s) per socket" | awk '{print $4}')
SOCKETS=$(lscpu | grep "^Socket(s)" | awk '{print $2}')
OPTIMAL_THREADS=$((CORES_PER_SOCKET * SOCKETS))

export OMP_NUM_THREADS=$OPTIMAL_THREADS
echo "Optimal threads: $OPTIMAL_THREADS"
```

### NUMA Affinity (For .metal or multi-socket instances)

```bash
# Check NUMA topology
numactl --hardware

# For large arrays (> 32 MB), use SPREAD affinity
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# For cache-friendly operations, use COMPACT
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

See [NUMA Affinity Guide](NUMA_AFFINITY_GUIDE.md) for advanced NUMA optimization.

### CPU Frequency Scaling

```bash
# Set to performance mode for consistent benchmarks
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Verify
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
```

---

## 📊 Production Best Practices

### 1. Create Systemd Service (For Long-Running Applications)

```bash
# Create service file
sudo tee /etc/systemd/system/hpcs-app.service << 'EOF'
[Unit]
Description=HPCSeries Application
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/app
Environment="OMP_NUM_THREADS=4"
Environment="PATH=/home/ec2-user/hpcs-env/bin:/usr/local/bin:/usr/bin"
ExecStart=/home/ec2-user/hpcs-env/bin/python /home/ec2-user/app/main.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable hpcs-app
sudo systemctl start hpcs-app

# Check status
sudo systemctl status hpcs-app
```

### 2. Monitoring with CloudWatch

**Install CloudWatch Agent:**

```bash
# Amazon Linux
sudo dnf install -y amazon-cloudwatch-agent

# Configure custom metrics
cat << 'EOF' | sudo tee /opt/aws/amazon-cloudwatch-agent/etc/config.json
{
  "metrics": {
    "namespace": "HPCSeries",
    "metrics_collected": {
      "cpu": {
        "measurement": [{"name": "cpu_usage_idle", "rename": "CPU_IDLE", "unit": "Percent"}],
        "totalcpu": false
      },
      "mem": {
        "measurement": [{"name": "mem_used_percent", "rename": "MEM_USED", "unit": "Percent"}]
      }
    }
  }
}
EOF

# Start agent
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config \
  -m ec2 \
  -c file:/opt/aws/amazon-cloudwatch-agent/etc/config.json \
  -s
```

### 3. Auto-Scaling Configuration

**Create AMI from configured instance:**

```bash
# From your local machine
aws ec2 create-image \
  --instance-id i-XXXXXXXXX \
  --name "hpcs-v0.7-calibrated-$(date +%Y%m%d)" \
  --description "HPCSeries v0.7 with calibration (CPU-optimized instance)"
```

**Create Launch Template and Auto-Scaling Group** (via AWS Console or CLI)

### 4. Application Deployment with CodeDeploy

```bash
# Install CodeDeploy agent
cd /tmp
wget https://aws-codedeploy-us-east-1.s3.us-east-1.amazonaws.com/latest/install
chmod +x ./install
sudo ./install auto

# Create appspec.yml in your repository
cat << 'EOF' > ~/app/appspec.yml
version: 0.0
os: linux
files:
  - source: /
    destination: /home/ec2-user/app
hooks:
  BeforeInstall:
    - location: scripts/stop_app.sh
      timeout: 300
  AfterInstall:
    - location: scripts/install_dependencies.sh
      timeout: 600
  ApplicationStart:
    - location: scripts/start_app.sh
      timeout: 300
EOF
```

---

## 💰 Cost Optimization

### 1. Use Spot Instances (70% Savings)

```bash
# Launch spot instance (adjust instance-type and MaxPrice)
aws ec2 run-instances \
  --image-id ami-XXXXXXXXX \
  --instance-type c6i.2xlarge \
  --instance-market-options 'MarketType=spot,SpotOptions={MaxPrice=0.15,SpotInstanceType=one-time}'
```

**Trade-off**: Can be interrupted if demand spikes (2-minute warning)

### 2. Use Reserved Instances (40-60% Savings)

For predictable workloads, purchase Reserved Instances or Savings Plans.

### 3. Stop Instances When Not in Use

```bash
# Stop instance (stops billing for compute)
aws ec2 stop-instances --instance-ids i-XXXXXXXXX

# Start when needed
aws ec2 start-instances --instance-ids i-XXXXXXXXX
```

### 4. Right-Size Your Instance

Monitor CPU and memory usage:

```bash
# Check average CPU utilization
aws cloudwatch get-metric-statistics \
  --namespace AWS/EC2 \
  --metric-name CPUUtilization \
  --dimensions Name=InstanceId,Value=i-XXXXXXXXX \
  --start-time 2025-01-01T00:00:00Z \
  --end-time 2025-01-13T00:00:00Z \
  --period 3600 \
  --statistics Average
```

If consistently < 40%, consider downsizing.

---

## 🔧 Troubleshooting

### Issue: Slow Performance After Deployment

**Check 1: Verify SIMD is active**

```bash
python -c "import hpcs; print(hpcs.simd_info())"
# Should show AVX-512 on c6i/c7i, AVX2 on c5
```

**Check 2: Run calibration**

```bash
python -c "import hpcs; hpcs.calibrate(); hpcs.save_calibration_config()"
```

**Check 3: Verify OpenMP threads**

```bash
echo $OMP_NUM_THREADS  # Should be set to physical core count
```

### Issue: Import Error

```bash
# Check if HPCSeries is installed
pip list | grep hpcs

# Reinstall if needed
pip install --force-reinstall hpcs
```

### Issue: Out of Memory

```bash
# Check memory usage
free -h

# Consider:
# 1. Upgrading instance type
# 2. Processing data in batches
# 3. Using masked operations for sparse data
```

---

## 📚 Next Steps

After successful deployment:

1. ✅ **Verify performance** - Run benchmarks on your actual data
2. ✅ **Set up monitoring** - CloudWatch metrics and alarms
3. ✅ **Configure auto-scaling** - Handle variable workloads
4. ✅ **Implement CI/CD** - Automated deployment pipeline
5. ✅ **Optimize costs** - Spot instances, right-sizing

---

## 📖 Related Documentation

- **[Performance Guide](source/user_guide/performance.rst)** - Optimization and tuning
- **[NUMA Affinity Guide](NUMA_AFFINITY_GUIDE.md)** - Multi-socket optimization
- **[Calibration Guide](CALIBRATION_GUIDE.md)** - Performance auto-tuning
- **[API Reference](source/api/index.rst)** - Complete function reference

---

## 🎉 Summary

**You now have:**

✅ **Production-ready HPCSeries deployment** on AWS EC2
✅ **Bare metal installation** for maximum performance
✅ **Performance calibration** for your specific instance type
✅ **Cost optimization** strategies
✅ **Monitoring and scaling** best practices

**Expected Performance (AVX-512 instances: c7i/c6i family):**
- Rolling median (1M elements): ~8-12 ms (100-150x faster than Pandas)
- SIMD reductions: 3-5x faster than NumPy
- Large-scale processing: Scales linearly with vCPU count

**Performance on other families:**
- **c7g (ARM)**: Comparable to AVX2, ~15-20 ms rolling median
- **c5 (AVX2)**: ~10-15 ms rolling median, 2-3x faster than NumPy

**Deployment time**: ~10-15 minutes from instance launch to production-ready

---

**Ready for production deployment!** 🚀
