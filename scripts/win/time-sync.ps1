# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
# Windows laiko sinchronizacija (paleisti PowerShell kaip Administrator).
# Kodėl: WSL VM laikas seka Windows laiką per Hyper-V PTP (/dev/ptp_hyperv); Windows laikrodis buvo +0,89 s nuo UTC,
# nes W32Time tarnyba nepaleista. Vienai mašinai tai nesvarbu, bet keliems serveriams (ir MXL Fabrics) visi host'ai
# privalo būti ant to paties laiko šaltinio. Serveriuose — PTP (ptp4l + phc2sys), žr. docs/architektura/2026-11-11-laikas-ir-sinchronizacija.md.
# Atšaukti: Set-Service W32Time -StartupType Manual; Stop-Service W32Time

Set-Service -Name W32Time -StartupType Automatic
Start-Service W32Time
w32tm /config /manualpeerlist:"time.windows.com,0x9 0.pool.ntp.org,0x9" /syncfromflags:manual /update
Restart-Service W32Time
w32tm /resync /force
Start-Sleep -Seconds 5
w32tm /query /status
w32tm /stripchart /computer:time.windows.com /samples:5 /dataonly
