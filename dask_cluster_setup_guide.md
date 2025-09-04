# How to Set Up a Dask Cluster

## Overview

This guide walks you through setting up a distributed Dask cluster across multiple machines. A Dask cluster consists of a scheduler that coordinates work and one or more workers that execute tasks.

---

## Prerequisites

> ⚠️ **Important**: All machines in the cluster must have the same working environment, including identical libraries and versions.

---

## Step 1: Check Machine IP Addresses

Before setting up the cluster, you need to identify the IP addresses of all machines that will participate in the cluster.

### Windows
```cmd
ipconfig
```

### Verify Network Connectivity

Test connectivity between machines using the ping command:

```cmd
ping IP_ADDRESS
```

Replace `IP_ADDRESS` with the actual IP address of each machine you want to include in the cluster.

Set network profile on "Private" on the machine you want to use to be able to communicate with other machines with the same network profile.

*Example: if one device has its network profile set to "Public" it won't be able to receive pings from another device which has its network profile set to "Private".*
https://www.lifewire.com/change-networks-to-private-on-windows-10-5185933
https://www.reddit.com/r/networking/comments/17an5a8/pc_not_pingable_from_same_subnet_but_i_can_ping/

---

## Step 2: Set Up the Scheduler

The scheduler is the central coordinator of your Dask cluster. Choose one machine to act as the scheduler.

### Command
```bash
dask scheduler SCHEDULER_MACHINE_IP_ADDRESS:8786
```

### Notes
- Replace `SCHEDULER_MACHINE_IP_ADDRESS` with the actual IP address of your scheduler machine
- Port `8786` is the default Dask scheduler port
- The scheduler machine can also function as a worker (see Step 3)

---

## Step 3: Set Up Workers

Workers are the machines that will execute the actual computations. Run this command on each machine you want to use as a worker.

### Command
```bash
dask worker tcp://SCHEDULER_MACHINE_IP_ADDRESS:8786
```

### Scheduler as Worker
The scheduler machine can also serve as a worker:

1. Open a new terminal on the scheduler machine
2. Run the same worker command shown above
3. The machine will connect to itself and function as both scheduler and worker

---

## Step 4: Run Your Code

Once your cluster is set up, you can start running your Dask code. Dask will automatically:

- Distribute tasks across all connected workers
- Handle load balancing
- Manage data movement between workers
- Coordinate results back to your client

### Example Usage
```python
import dask
from dask.distributed import Client

# Connect to your cluster
client = Client('SCHEDULER_MACHINE_IP_ADDRESS:8786')

# Your Dask code here
# Dask will automatically distribute the work
```

---

## Environment Requirements

### Critical Requirement
🔧 **All cluster machines must have identical environments:**

- Same Python version
- Same library versions
- Same dependencies
- Compatible operating systems

### Verification Checklist
- [ ] Python versions match across all machines
- [ ] Dask versions are identical
- [ ] All required libraries are installed with matching versions
- [ ] Network connectivity is established between all machines
- [ ] Firewall settings allow communication on port 8786

---

## Troubleshooting

### Common Issues
- **Connection refused**: Check firewall settings and ensure port 8786 is open
- **Version mismatch**: Verify all machines have identical library versions
- **Network issues**: Confirm machines can ping each other successfully

### Monitoring
- The Dask dashboard is available at `http://SCHEDULER_MACHINE_IP_ADDRESS:8787`
- Use this interface to monitor cluster status, task progress, and resource usage

---

## Summary

Your Dask cluster setup is complete when:
1. ✅ All machine IPs are identified and reachable
2. ✅ Scheduler is running on the designated machine
3. ✅ All workers are connected to the scheduler
4. ✅ Your code can connect to the cluster and submit tasks

The cluster will now automatically handle distributed computing for your Dask workflows!