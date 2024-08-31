---
tags:
  - 读卡器
---

# NFC
## FAQ
### IC卡和ID卡的区别
IC卡是可读可写可加密，具有16个扇区，在使用扩展性上可以使用在16个场景之上，而且还能存储信息，进行数据交互，而ID卡则只有一个卡号而已，没有存储功能，在使用上局限性很大，要使用也只能使用ID卡的卡号，无其他任何可以用，不能写，不能加密，只能读。 所以IC卡是具有存储功能，而ID卡则无。

## IC卡
IC卡全称集成电路卡（Integrated Circuit Card），又称智能卡（Smart Card）它是将一个微电子芯片嵌入符合ISO 7816标准的卡基中，做成卡片形式，芯片含的存储器（ROM、EEPROM）、保护逻辑电路、甚至带微处理器CPU。带有CPU的IC卡才是真正的智能卡。可读写，容量大，有加密功能，数据记录可靠，使用更方便。
按照嵌入集成电路芯片的形式和芯片类型的不同IC卡大致可分为接触式，非接触式、双界面卡。IC卡可以分为:接触式IC卡、非接触式IC卡和双界面IC卡，而非接触式IC卡简介又称射频卡，成功地解决了无源(卡中无电源)和免接触这一难题，是电子器件领域的一大突破。

IC卡的不同类型和型号各自有不同的功能和应用。以下是对您提到的各个卡片的简要说明：

1. CPU卡
CPU卡是一种智能卡，内置中央处理单元，能够进行复杂的计算和数据处理。它们通常具有更高的安全性和功能，适用于金融、身份验证等领域。

2. SAM卡
SAM（Security Access Module）卡是一种用于增强安全性的智能卡，主要用于加密和解密操作。它们常用于支付系统和安全通信中，提供密钥管理和数据保护。

3. SLE4442
SLE4442是一种非接触式的智能卡，常用于身份识别和门禁控制。它具有4K位的EEPROM存储，支持多种数据传输协议。

4. SLE4428
SLE4428类似于SLE4442，但通常具备更小的存储容量（2K位的EEPROM）。它也用于身份识别和简单的存储应用。

5. AT24Cxx
AT24Cxx是一系列EEPROM芯片，通常用于存储小量数据。它们支持I2C接口，广泛应用于各种电子设备中，如配置存储和数据记录。

IC复制卡种类和功能说明：
- UID卡：最常见的IC复制卡，可以反复擦写
- CUID卡：防火墙IC复制卡，可以通过70%的防火墙系统，不锁卡
- FUID卡：防火墙IC复制卡，可以通过90%的防火墙系统，不锁卡
- UFUID卡：防火墙IC复制卡，可以通过90%的防火墙系统，不锁卡

<table>
  <tr>
    <th>卡类型</th>
    <th>芯片名称</th>
    <th>频率</th>
    <th>擦写</th>
    <th>说明</th>
  </tr>
  <tr>
    <td rowspan="7">IC卡</td>
    <td>M1卡</td>
    <td>13.56MHz</td>
    <td>❌</td>
    <td>普通卡，0块（卡号块）出厂为锁定状态，不可用于复制，只能授权，物业卡均为此卡，俗称授权卡</td>
  </tr>
  <tr>
    <td>UID卡</td>
    <td>13.56MHz</td>
    <td>✔</td>
    <td>经典复制卡，“魔术卡”，可以反复擦写</td>
  </tr>
  <tr>
    <td>CUID卡</td>
    <td>13.56MHz</td>
    <td>✔</td>
    <td>防火墙IC复制卡，对于UID卡复制成功刷卡无效的情况下，可以使用此卡复刻</td>
  </tr>
  <tr>
    <td>FUID卡</td>
    <td>13.56MHz</td>
    <td>❌</td>
    <td>防火墙IC复制卡，对于CUID卡复制成功刷卡无效的情况下，可以使用此卡复刻，一次性写入锁卡</td>
  </tr>
  <tr>
    <td>FUID卡</td>
    <td>13.56MHz</td>
    <td>❌</td>
    <td>功能和FUID卡一样，支持手动锁卡，没有锁卡之前就是一张UID卡，锁卡后不支持擦写</td>
  </tr>
  <tr>
    <td>GTU卡</td>
    <td>13.56MHz</td>
    <td>✔</td>
    <td>可以自动复位数据，用于滚动码防复制系统，支持反复擦写</td>
  </tr>
  <tr>
    <td>GDMIC卡</td>
    <td>13.56MHz</td>
    <td>✔</td>
    <td>可以自动复位数据，用于滚动码防复制系统，支持反复擦写</td>
  </tr>
  <tr>
    <td rowspan="2">ID卡</td>
    <td> T5577卡<br>EM4305卡<br>5200卡 </td>
    <td> 125kHz </td>
    <td>✔</td>
    <td></td>
  </tr>
  <tr>
    <td> F8268卡 </td>
    <td> 125kHz </td>
    <td>✔</td>
    <td>防火墙版本的ID复制卡，用于57卡复制刷卡无效的情况</td>
  </tr>
</table>

滚动码是指，部分系统在刷卡一次后会将卡内某数据增加。

## RFID卡
RFID射频卡是结合RFID射频识别技术的一种非接触式IC卡，属于IC卡。它具有数据处理及安全认证功能等特有的优点，诞生于90年代初，是世界上最近几年发展起来的一项新技术。它能够成功地将射频识技术和IC卡技术结合起来，解决了无源（卡中无电源）和免接触这一难题，是电子器件领域的一大突破。

射频卡（RFID卡）中的不同类型和标准有特定的应用和特点。以下是您提到的各个卡片的简要说明：

1. Mifare S50
Mifare S50是一种常见的非接触式智能卡，具有1KB的EEPROM存储。它广泛用于公共交通、门禁系统和小额支付等场景。

2. Mifare S70
Mifare S70是Mifare系列的升级版，提供了更大的存储容量（4KB EEPROM）。它适用于需要更多数据存储的应用，如会员管理和复杂的身份验证。

3. Mifare Ultralight
Mifare Ultralight是一种成本较低的非接触式卡，通常用于一次性票务和短期应用。它的存储容量较小，适合简单的数据存储需求。

4. Type A CPU
Type A CPU卡是遵循ISO/IEC 7816标准的智能卡，使用Type A的通信协议。这类卡片通常用于安全性要求较高的应用，如金融交易和身份验证。

5. Type B CPU
Type B CPU卡同样遵循ISO/IEC 7816标准，但使用Type B的通信协议。它们适用于需要高安全性和加密功能的应用，如政府身份证和电子护照。

这些卡片各自具有不同的特性和适用场景，广泛应用于安全、支付和身份识别等领域。


## 数据
### Mifare Classic 1k
M1卡存储容量为1k， 被划分为16个扇区，每个扇区4个区块，每个区块16个字节。

0扇区为公共扇区

其中第1个区块存储卡片有关信息
- 1-4字节：卡号
- 5：卡号异或值，BCC校验信息，将卡号进行异或运算得出来的值
- 6：卡类型，08代表S50卡，20代表CPU卡，28代表CPU模拟卡
- 7-8：应答类别，0400通常为S50卡，0200为S70
- 9-16：生产厂商信息

第4个区块是控制区块

1扇区-15扇区是数据扇区：
- 每一个扇区可以加不同密钥
- 多个扇区存储不同数据，可实现一卡通

1-3区块是数据存储区
4区块是控制区块，前6个字节是密钥A，中间4个是存取控制位，后6个是密钥B

如果16个扇区的AB密钥都是`FF`，那么为非加密卡，如果部分密钥不是`FF`，那么就是半加密卡。

破解主要针对半加密卡。

示例数据：
- 第1区块：0C 08 21 05 01 23 00 03 FF FF 00 00 00 00 00 00
- 第2区块：00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
- 第3区块：00 00 00 00 00 00 00 00 FF FF FF FF FF FF 3B BE
- 第4区块：FF FF FF FF FF 99 FF 07 80 69 FF FF FF FF FF 99

## 工作原理
读卡器释放电磁波，卡内部有相同频率的协振电路，然后给电容充电。

卡片状态：
Power-Off --> Idle --> Ready --> Activate --> Halt

### ISO14443协议（标准）
#### ISO14443-A

常见：Mifare卡

#### ISO14443-B
- TypeB型卡片
- REQB命令请求
- ATQB应答
常见：二代身份证

## CX522双频刷卡器
<style>
    .highlight-pink {
        color: pink;
    }
    .highlight-red {
        color: red;
    }
    .highlight-blue {
        color: blue;
    }
    .highlight-green {
        color: green;
    }
    .highlight-skyblue {
        color: #00BFFF;
    }
    .highlight-barbiepink {
        color: #DA1D81;
    }
    .highlight-purple {
        color: purple;
    }
    .highlight-orange {
        color: orange;
    }
</style>

功能特点
1. 支持通过命令修改串口波特率，掉电保存，只需设置一次。
2. 默认串口设置：9600、8、N、1。
3. 检测IC卡靠近后：支持主动输出卡片序列号，可通过命令配置关闭，掉电保存，只需设置一次。**默认：检测到卡片序列号后主动输出**。
4. 检测到IC卡靠近后：当卡片离开的时候，输出固定的命令表示卡片离开。只有在主动输出模式下有效。**默认：关闭，需通过命令配置打开，掉电保存。只需配置一次**。
5. 每个模块上面都有指示灯，用于指示模块当前的工作的状态。a、当模块正常供电后，模块上的指示灯会闪烁一下；b、当模块的串口接收到数据后，会闪烁一下；c、当有效卡片靠近模块后，模块上的指示灯会常亮。当卡片离开后，指示灯熄灭。
6. 模块支持通过命令设置地址，地址范围从0x00-0xFE。掉电保存，只需设置一次，默认0x00。
7. 模块支持通过命令查询模块当前地址是多少。
8. 如果模块上面有蜂鸣器的话。可以通过命令控制蜂鸣器“滴”的响一声。
9. 刷卡蜂鸣器提醒，支持通过命令设置进行关闭。默认：打开，掉电保存，只需设置一次。
10. 模块支持命令寻卡、并实时反馈是否有卡片存在的状态。
11. 模块支持UART(TTL)口、485口、232口、USB口、韦根26、韦根34。在出货的时候只能选择其中的一种出货。不同型号的模块接口有所差异。
12. 支持输出格式的设定。

### 串口设置
波特率：9600；校验位：None; 数据位：8；停止位：1；

### 读卡具体的实现
第1步：刷卡。模块上的指示灯会亮，并且通过串口主动发送卡片序列号给上位机，此时上位机就会接收到卡片序列号的相关数据。

一张白卡主动输出卡片序列号的数据为：
20 00 00 08 04 00 00 00 97 42 8C 7B D1 03

用导顿的软件解析后发现其卡号为：2537720955
扇区0: 97 42 8C 7B 22 08 04 00 03 27 D2 87 DA AC 9F 90

- 20: 起始符
- 00：地址
- 00：命令字节，模块主动输出卡片序列号时，该字节为0x00；其他指令为命令字
- 08：表示后面8个字节为有效数据位（IC卡固定为8字节）
- 04 00：表示卡片属性为S50卡
- 00 00：此2个字节无实际意义。
- 97 42 8C 7B：表示卡片序列号。刷不同卡片，此4个字节会变。
- D1：校验和。从地址（SEQNR）开始到数据（DATA）的最后一字节异或，然后再取反 得到。
- 03： 帧结束符。

Python程序示例，端口号查询方式可参考【Log】-【Linux环境】-【串口】
```python
import serial


def servo_mode():
    # 打开串口
    ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)

    start_delimiter = b'\x20'
    end_delimiter = b'\x03'
    buffer = bytearray()

    try:
        while True:
            # 读取数据
            data = ser.read(1)

            if data:
                buffer += data

                # 如果遇到起始符，清空缓存并开始新的帧
                if buffer[0:1] != start_delimiter:
                    buffer.clear()
                    buffer += data

                # 如果遇到结束符，处理整个帧
                if buffer.endswith(end_delimiter):
                    # 打印完整帧，包括起始符和结束符
                    frame = " ".join(format(b, "02x") for b in buffer)
                    print(f"接收到的完整帧: {frame}")

                    # 清空缓存以等待下一个数据帧
                    buffer.clear()

    except KeyboardInterrupt:
        print("程序终止")
    finally:
        ser.close()


if __name__ == "__main__":
    servo_mode()

```

第2步：上位机就可以直接通过读指令、或者写指令、或者加密指令对IC卡进行读、或者写、或者加密操作了。
读块指令举例：读第2块数据，
上位机发送指令： 20 00 <span class="highlight-barbiepink">22</span> <span class="highlight-red">08</span> <span class="highlight-orange">00</span> FF FF FF FF FF FF <span class="highlight-green">02</span> <span class="highlight-blue">D7</span> 03

- 其中，粉色的<span class="highlight-barbiepink">22</span>表示：此条命令为22号指令。22号指令为读块指令；
- 其中，红色的<span class="highlight-red">08</span>表示：紧接着后面有效数据有8个字节；
- 其中，橙色的<span class="highlight-orange">00</span>表示：表示采用密码A的方式校验卡片的密码，若需要采用密码B的方式校验卡片的密码，则此字节改为0x01即可。注意：这个密码必须和需要读取的IC卡所在块对应的这个块所在扇区的密码A或者密码B 保持一样，否则就会读卡失败。
- 其中，绿色的<span class="highlight-green">02</span>表示：需要读取卡片的2号块数据；
- 其中，蓝色的<span class="highlight-blue">D7</span>为ECC校验码，是00 22 08 00 FF FF FF FF FF FF 02  首先异或，然后再取反得到的。异或运算有小工具可以用，见我们的资料中的《按位异或工具》文件夹下。

- 读块成功，则上位机收到第2块的数据，如下：
20 00 <span class="highlight-barbiepink">22</span> <span class="highlight-red">11</span> <span class="highlight-skyblue">00</span> 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 <span class="highlight-green">CC</span> 03
  - 其中，粉色的<span class="highlight-barbiepink">22</span>表示：该指令是读指令22号指令的返回指令。
  - 其中，红色的<span class="highlight-red">11</span>表示：紧接着后面有效数据有17个字节；
  - 其中，青色的<span class="highlight-skyblue">00</span>表示：此读卡指令执行成功；
  - 其中，绿色的<span class="highlight-green">CC</span>表示：ECC校验码；是00 22 11 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  首先异或，然后再取反得到的。
- 读块失败，若是校验密码失败上位机收到指令：20 00 22 01 01 DD 03
其中，红色的22表示：该指令是读指令22号指令的返回指令。
其中，绿色的01表示：紧接着后面有效数据有1个字节；
其中，蓝色的01表示：次读卡指令执行失败，原因—密码校验失败。
所有的读写操作过程中，只要密码校验失败都会反馈这条指令给上位机。
- 读块失败，若是没检测到卡片上位机收到指令：20 00 <span class="highlight-red">22</span> <span class="highlight-green">01</span> <span class="highlight-skyblue">02</span> DE 03
  - 其中，红色的<span class="highlight-red">22</span>表示：该指令是读指令22号指令的返回指令。
  - 其中，绿色的<span class="highlight-green">01</span>表示：紧接着后面有效数据有1个字节；
  - 其中，蓝色的<span class="highlight-skyblue">02</span>表示：没有检测到卡片。
- 读块失败，若是块读取失败 上位机收到指令： 20 00 22 01 04 D8 03

```python
import serial
import time


def xor_and_invert(data):
    # 初始化结果
    result = 0x00

    # 对所有字节进行异或操作
    for byte in data:
        result ^= byte

    # 取反
    result = ~result & 0xFF  # 使用 & 0xFF 保证结果在0x00到0xFF之间

    return result


def read_data(block_id, sector=None, verbose=True):
    if sector is not None:
        if block_id > 3:
            Warning("错误，指定扇区后。区块ID不能大于3")
            return None

        total_id = block_id + sector * 4
    else:
        total_id = block_id

    # 打开串口
    ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)

    data_send = [0x00, 0x22, 0x08, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, total_id]
    ECC = xor_and_invert(data_send)

    # 要发送的指令
    command = bytes([0x20] + data_send + [ECC] + [0x03])

    # 发送指令
    ser.write(command)
    frame = " ".join(format(b, "02x") for b in command)
    if verbose:
        print(f"发送的指令: {frame}")

    start_delimiter = b'\x20'
    end_delimiter = b'\x03'
    buffer = bytearray()

    try:
        while True:
            # 读取数据
            data = ser.read(1)

            if data:
                buffer += data

                # 如果遇到起始符，清空缓存并开始新的帧
                if buffer[0:1] != start_delimiter:
                    buffer.clear()
                    buffer += data

                # 如果遇到结束符，处理整个帧
                if buffer.endswith(end_delimiter):
                    # 打印完整帧，包括起始符和结束符
                    frame_data = [format(b, "02x") for b in buffer]
                    frame = " ".join(frame_data)
                    if verbose:
                        print(f"接收到的完整帧: {frame}")

                    if frame_data == ['20', '00', '22', '01', '02', 'de', '03']:
                        print("没有卡")
                        block_data = None  # 没有卡
                    else:
                        data_len = buffer[3]
                        block_data = frame_data[5:4+data_len]

                    # 完成一个帧后跳出循环
                    break

    except KeyboardInterrupt:
        print("程序终止")
    finally:
        ser.close()

    return block_data

if __name__ == "__main__":
    for i in range(16):
        for j in range(4):
            print("\n扇区%d，区块%d" % (i, j), end=': ')
            data_block = read_data(j, i, verbose=False)
            print(" ".join(data_block), end='')
            time.sleep(0.08)
        print()

```

#### 寻卡
上位机发送：`20 00 27 00 D8 03`
- 情况1：若无卡片存在，则上位机收到无卡片的指令：`20 00 27 01 02 DB 03`
- 情况2：若有卡片存在，则上位机收到卡片相关指令：`20 00 00 08 04 00 00 00 A6 40 FE E4 0E`

例如：接收到数据

20 00 <span class="highlight-barbiepink">27</span> <span class="highlight-red">08</span> <span class="highlight-skyblue">04</span> 00 00 00 97 42 8c 7b <span class="highlight-green">f6</span> 03

- 其中，粉色的<span class="highlight-barbiepink">27</span>表示：该指令是读指令27号指令的返回指令。
- 其中，红色的<span class="highlight-red">08</span>表示：紧接着后面有效数据有8个字节；
- 其中，蓝色的<span class="highlight-skyblue">04</span>表示：；
- 其中，绿色的<span class="highlight-green">f6</span>表示：ECC校验码；是00 27 08 04 00 00 00 97 42 8c 7b。首先异或，然后再取反得到的。

```python
def exist_card():
    # 打开串口
    ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)

    # 要发送的指令
    command = bytes([0x20, 0x00, 0x27, 0x00, 0xD8, 0x03])

    # 发送指令
    ser.write(command)
    frame = " ".join(format(b, "02x") for b in command)
    print(f"发送的指令: {frame}")

    start_delimiter = b'\x20'
    end_delimiter = b'\x03'
    buffer = bytearray()

    exist = False
    try:
        while True:
            # 读取数据
            data = ser.read(1)

            if data:
                buffer += data

                # 如果遇到起始符，清空缓存并开始新的帧
                if buffer[0:1] != start_delimiter:
                    buffer.clear()
                    buffer += data

                # 如果遇到结束符，处理整个帧
                if buffer.endswith(end_delimiter):
                    # 打印完整帧，包括起始符和结束符
                    frame_data = [format(b, "02x") for b in buffer]
                    frame = " ".join(frame_data)
                    print(f"接收到的完整帧: {frame}")

                    if frame_data != ['20', '00', '27', '01', '02', 'db', '03']:
                        if frame_data[4:6] == ['04', '00']:
                            print("检测到S50卡，卡号：", " ".join(frame_data[-6: -2]))
                        else:
                            print("检测到未知卡，卡号：", " ".join(frame_data[-6: -2]))
                        exist = True

                    # 完成一个帧后跳出循环
                    break

    except KeyboardInterrupt:
        print("程序终止")
    finally:
        ser.close()

    return exist

```

### 写卡具体的实现
写块指令举例： 在第2块写入：数据。


上位机发送指令：20 00 <span class="highlight-barbiepink">23</span> 18 00 FF FF FF FF FF FF 02 00 11 22 33 44 55 66 77 88 99 aa bb cc dd ee ff C6 03
- 其中，粉色的<span class="highlight-barbiepink">23</span>表示：此条命令为23号指令。23号指令为写块指令；
- 其中，红色的18表示：紧接着后面有效数据有24个字节；
- 其中，青色的00表示：表示采用密码A的方式校验卡片的密码；
- 其中，紫色的FF FF FF FF FF FF表示：此条写块指令采用的密码；注意：这个密码必须和需要读取的IC卡所在块对应的这个块所在扇区的密码A或者密码B 保持一样，否则就会写卡失败。
- 其中，绿色的02表示：需要把后面的16个字节写到卡片的2号数据块；
- 其中，00 11 22 33 44 55 66 77 88 99 aa bb cc dd ee ff 表示：要写入到IC中的数据
- 其中，蓝色的C6为ECC校验码，是00 23 18 00 FF FF FF FF FF FF 02 00 11 22 33 44 55 66 77 88 99 aa bb cc dd ee ff首先异或，然后再取反得到的。

- 写块成功，则上位机收到指令： 20 00 23 01 00 DD 03
  - 其中，红色的23表示：该指令是读指令23号指令的返回指令。
  - 其中，蓝色的01表示：后面有效字节1字节。
  - 其中，绿色的00表示：写指令执行成功。
- 写块失败，若是校验密码失败上位机收到指令：20 00 23 01 01 DC 03
- 写块失败，若是没检测到卡片上位机收到指令：20 00 23 01 02 DF 03
- 写块失败，若是块写入失败 上位机收到指令： 20 00 23 01 04 D9 03


## 常见问题
- 身份证会“消磁”吗？

目前二代身份证是“非接触式IC卡”的芯片结构。非接触式IC卡是由IC芯片、感应天线组成，封装在一个标准的PVC卡片内，芯片及天线无任何外露部分。卡片在一定距离范围（通常为5至10厘米）靠近二代证阅读器表面，通过无线电波的传递来完成数据的读写操作。

“简单点说，身份证是靠电磁波工作，虽然有个‘磁’，但跟磁场没有什么关系。”民警介绍称，在生活中，确实会出现“身份证一直读不出来”的现象。从身份证的芯片结构来看，导致身份证失效、无法读取的情况一般是卡片内的天线（线圈）脱焊或者出现其他物理损坏以及卡中的芯片坏掉造成的。

“身份证不怕水，也可抵御正常高温。但是如果用力扭曲、重压或放置在高温环境中，会造成身份证内芯片坏掉。”他说，超强的磁场、辐射（如医院的X光机）等因素确实可能会导致线圈产生异常电荷运动，损害身份证中芯片（并非“消磁”），但是一般手机磁场、安检机器、常见磁铁等不会损害身份证。

所以，当你的身份证无法使用的时候，不是“消磁”了。有可能是天线脱焊或者是芯片损坏了，平时折弯过身份证或者是用力扭曲过身份证。这种情况下，你就去补办一个吧！


