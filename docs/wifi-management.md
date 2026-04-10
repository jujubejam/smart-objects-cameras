# WiFi Network Management

> **Prefer slides?** View the [WiFi Management Slides](https://kandizzy.github.io/smart-objects-cameras/wifi-management-slides.html) for a visual walkthrough.

Managing WiFi connections on your Raspberry Pi cameras when moving between different networks (home ↔ classroom).

---

## Switching Between Networks (Home ↔ Classroom)

Modern Raspberry Pi OS uses **NetworkManager** for WiFi configuration. Here are the best methods:

### Method 1: Using nmtui (Recommended - Menu Interface)

**Perfect for adding classroom WiFi before you get to class!**

`nmtui` is a text-based menu interface that lets you add networks you're not currently connected to:

```bash
# SSH into the Pi
ssh orbit

# Open NetworkManager Text UI
sudo nmtui
```

**Steps in nmtui:**
1. Select **"Activate a connection"** to see current networks, OR
2. Select **"Edit a connection"** to add new networks
3. To add a new WiFi network:
   - Choose **"Add"**
   - Select **"Wi-Fi"**
   - Enter SSID: `ClassroomWiFi`
   - Enter Password: `classpassword`
   - Tab to **"OK"** and press Enter
   - Tab to **"Back"** and press Enter
4. Select **"Quit"**

**Tip:** You can add as many networks as you want! The Pi will automatically connect to any available network you've saved.

---

### Method 2: Using nmcli (Command Line)

Quick commands for adding and switching networks:

```bash
# List available WiFi networks
nmcli device wifi list

# Connect to a network (saves automatically - but requires network to be in range!)
sudo nmcli device wifi connect "ClassroomWiFi" password "classpassword"

# List saved connections
nmcli connection show

# Switch to a previously saved network
nmcli connection up "ClassroomWiFi"

# Delete a saved network
sudo nmcli connection delete "OldNetwork"
```

**Note:** `nmcli device wifi connect` only works if the network is currently in range. To add a network you're not near, use `nmtui` instead!

---

### Method 3: Using raspi-config (Graphical Alternative)

If you prefer a menu interface:

```bash
sudo raspi-config
# Navigate: System Options → Wireless LAN
# Enter new SSID and password
# Reboot
```

---

### Method 4: Check Current WiFi Settings

```bash
# See which network you're connected to
nmcli device wifi

# See detailed connection info
nmcli connection show --active

# See WiFi password for saved network
sudo nmcli connection show "YourNetworkName" | grep psk
```

---

### Emergency: Editing via SD Card (If Locked Out)

If you can't connect to the Pi at all:

1. Power off the Pi and remove SD card
2. Insert SD card into your computer
3. In the `boot` partition, create a file called `wpa_supplicant.conf`:

   ```
   country=US
   ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
   update_config=1

   network={
       ssid="ClassroomWiFi"
       psk="classpassword"
   }
   ```

4. Save, eject SD card, and boot the Pi
5. Once connected, the Pi will import this to NetworkManager

---

## Related Documentation

- [README.md](../README.md) - Main documentation
- [Multi-User Access](multi-user-access.md) - Collaborative work on shared Pis
