# 安卓 WebView / APK 可行性核查（事实核查简报）

约定：**【官方】**＝厂商/标准组织文档或代码仓库原文；**【第三方】**＝博客、论坛、Issue，未经官方背书。

## 一、不装 Android Studio 做出 WebView APK

**1）现成套壳 App** — 【官方】Hermit 帮助页明确写「Google Play 政策不允许应用在设备上生成 APK」，它只能建 "Lite App" 主屏图标（[hermit.chimbori.com/help](https://hermit.chimbori.com/help/)）。Native Alpha（GPL，Play/IzzyOnDroid/GitHub 可下，Android 8+）同样只用系统 WebView 生成全屏快捷方式（[README](https://github.com/cylonid/NativeAlphaForAndroid)）。"WebView Browser Tester" 一类是 WebView 渲染/兼容测试器，不是套壳生成器。→ 能给你"像 App 的图标"，但每个同学要先装同一宿主 App，图标还依赖桌面（Hermit FAQ 提到三星/小米桌面快捷键失效）。自用可以，**不适合"发个文件给同学"**。

**2）云构建** — 【第三方，厂商博客】code2native 明确要求「URL 必须已上线且为 HTTPS 公网域名」，排除 localhost 与内网地址（[指南](https://code2native.com/blog/url-to-apk)）。PWABuilder/GoNative/Median 走 PWA/TWA 路线，【官方】web.dev 说明可安装性以 HTTPS 安全上下文为前提（[怎样才算可安装](https://web.developers.google.cn/articles/install-criteria?hl=zh-cn)、[PWA 上架应用商店](https://web.developers.google.cn/articles/pwas-in-app-stores?hl=zh-cn)）。**局域网 `http://192.168.x.x:8000` 一律走不通。**

**3）GitHub Actions** — 【官方】GitHub 托管 ubuntu runner 镜像预装 Android SDK（[actions/runner-images](https://github.com/actions/runner-images)）。最小工程＝一个 AGP 工程＋单个 WebView Activity：

```yaml
on: [workflow_dispatch]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-java@v4
        with: {distribution: temurin, java-version: '17'}
      - run: chmod +x gradlew && ./gradlew assembleDebug
      - uses: actions/upload-artifact@v4
        with: {name: app-debug, path: app/build/outputs/apk/debug/app-debug.apk}
```

产物在 Actions 运行页 Artifacts 下载（需 GitHub 账号与仓库权限）。现成模板：【第三方】[henryhale/android-capacitor](https://github.com/henryhale/android-capacitor)。本机零 SDK，但要会写 Kotlin/Java 与 Gradle。

**4）website-to-apk** — 【官方仓库】MIT、213★、**仍在维护**（2026-03 仍有提交，未归档）：自动下载 JDK 17 与 Android SDK 工具，无需 Android Studio；`mainURL` 可直接写 `http://192.168.x.x:8000`；自带 `allowMixedContent`、`trustUserCA` 等选项（[README](https://github.com/Jipok/website-to-apk)）。三个坑：脚本是 bash（`./make.sh`，**Windows 需 Git Bash/WSL**）；仍要下数百 MB~1GB SDK；模板是否默认放行明文 HTTP 需自己改 manifest。

**5）Capacitor/Cordova** — 【官方】CLI 支持 `npx cap build android --androidreleasetype APK` 命令行出包（[cap build](https://capacitorjs.com/docs/cli/commands/build)），但前提是 JDK 17＋SDK 的 cmdline-tools/platform-tools/build-tools/platform（约 0.5–1GB），且文档整体面向 Android Studio。对比 Android Studio 官方下载约 1.1GB（[developer.android.com/studio](https://developer.android.com/studio)）。没有"必需组件最小集"官方页面。

## 二、WebView 访问局域网 HTTP / HTTPS

**1）明文 HTTP** — 【官方】targetSdk 28+ 默认拒绝明文（[网络安全配置](https://developer.android.com/privacy-and-security/security-config)）。全局写法：`<application android:usesCleartextTraffic="true">`；**推荐只放行一个 IP**：

```xml
<!-- res/xml/network_security_config.xml -->
<network-security-config>
  <base-config cleartextTrafficPermitted="false"/>
  <domain-config cleartextTrafficPermitted="true">
    <domain includeSubdomains="false">192.168.137.1</domain>
  </domain-config>
</network-security-config>
```
manifest：`<application android:networkSecurityConfig="@xml/network_security_config">`。配了 NSC 时以 NSC 为准；否则报 `net::ERR_CLEARTEXT_NOT_PERMITTED`。

**2）自签名 HTTPS** — 【官方】`WebViewClient.onReceivedSslError` 默认取消加载 → 白屏/错误页（[WebViewClient](https://developer.android.com/reference/android/webkit/WebViewClient)）。要在回调里比对 host 等于你电脑 IP 才 `handler.proceed()`，否则 `cancel()`；Google 官方 FAQ 专门警告此类处理的安全风险（[Google 支持页](https://support.google.com/faqs/answer/7071387)）。另一条路是让 App 信任用户安装的 CA（Android 7+ 默认不信任，需 NSC trust-anchors）——website-to-apk 的 `trustUserCA` 就是干这个。

**3）CORS** — 【官方】MDN：CORS 只支持 http/https，现代浏览器把本地文件当 **opaque origin（null）**，`file:///android_asset/index.html` 里 fetch 你的接口**会报错**（[MDN: CORS request not HTTP](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/CORS/Errors/CORSRequestNotHttp)），而你现有服务没加任何头。服务端至少要 `Access-Control-Allow-Origin: *`，POST JSON 还会触发 preflight，需要 Allow-Methods/Allow-Headers。官方推荐不用 file://，改用 `WebViewAssetLoader`＋`https://appassets.androidplatform.net`（[加载应用内内容](https://developer.android.com/develop/ui/views/layout/webapps/load-local-content)），但那**不消除跨源**，只是换成 https origin，CORS 照旧还叠加混合内容限制。**最省事：让 WebView 直接加载 `http://192.168.x.x:8000` 本身**——同源、无 CORS、无证书问题。

**4）混合内容** — 【官方】`setMixedContentMode` 默认 `MIXED_CONTENT_NEVER_ALLOW`，HTTPS 页面里的 HTTP 请求被阻断，需显式改 `ALWAYS_ALLOW`（[WebSettings](https://developer.android.com/reference/android/webkit/WebSettings)）。

## 三、手机如何找到电脑

1. **二维码**：标准库没有二维码生成（`qrcode` 是第三方）。零依赖做法是用在线生成器把 URL 转 PNG 打印；只解决"输入"，解决不了"IP 会变"。
2. **mDNS/Bonjour**：**标准库做不到**。Python 无 mDNS/DNS-SD 实现（第三方如 [python-zeroconf](https://github.com/python-zeroconf/python-zeroconf)），手写 socket 多播等于重写库，与"零第三方依赖"硬约束冲突，建议放弃。Android 端只有 App 内 `NsdManager` 能发现服务（[NsdManager](https://developer.android.com/reference/android/net/nsd/NsdManager)），系统级 `.local` 解析至今是未实现的开放请求（[issuetracker 140786115](https://issuetracker.google.com/issues/140786115)），浏览器输 `xxx.local:8000` 不可靠。
3. **静态 IP / DHCP 保留**：Windows 改 IPv4 属性约 2 分钟；更稳的是路由器按 MAC 绑定固定 IP（【第三方】[PCMag](https://www.pcmag.com/how-to/how-to-set-up-a-static-ip-address)）。难度低，各路由器界面不同。
4. **手机热点**：手机做 AP 时电脑通常拿 192.168.43.x 一类地址，**并不更稳定**。反过来让 Windows 开"移动热点"（ICS），电脑自己当网关，传统默认 192.168.137.1（【第三方】资料，Microsoft 官方未写死该地址），IP 固定、不需要路由器，演示最可控。

## 四、两条路线

**路线 A（推荐，零构建）**：什么都不装（或手机装 Native Alpha/Hermit）；产物＝主屏图标＋二维码。Windows 可行性 100%。最大风险：IP 变了图标失效（固定 IP / 电脑热点即可消除）。**成本最低、现场最不易翻车。**

**路线 B（自建 APK）**：GitHub Actions（本机零 SDK）或 website-to-apk（自动下 JDK/SDK，需 Git Bash）；产物＝能发出去的 APK。风险：改 manifest 放行明文、签名与更新、Play Protect 警告，外加两个 2026 新变量——开发者验证正逐步覆盖侧载（【官方】[Android 博客](https://developer.android.google.cn/blog/posts/android-developer-verification-rolling-out-to-all-developers-on-play-console-and-android-developer-console)），以及 **Android 17 起"局域网保护"会静默阻断对局域网地址的 TCP 连接，除非应用持有 ACCESS_LOCAL_NETWORK/NEARBY_DEVICES**（【官方】[本地网络权限](https://developer.android.com/privacy-and-security/local-network-permission)；【第三方】实证 [Mozilla Bug 2053432](https://bugzilla.mozilla.org/show_bug.cgi?id=2053432)、[Kodi #28557](https://github.com/xbmc/xbmc/issues/28557)）。这让"发 APK 给同学"越来越难。

**APK 比浏览器书签多了什么？** 引擎永远在电脑上时，APK ≈ 全屏、无地址栏、带图标的 WebView：换来图标感和"能传一个文件"，换不来离线、本地模型、推送；一离开局域网，它和书签一样是死页面。**结论：几乎没有。**

## 给用户的直白结论

1. 别为这个 demo 折腾 APK：引擎在电脑上，APK 只是把同一个网页装进壳里，价值几乎为零。
2. 想要"App 感"，就手机浏览器打开后"添加到主屏幕"，或用 Native Alpha 生成全屏图标，0 下载、0 构建。
3. 真要发给同学，就用最小 Gradle 工程＋GitHub Actions 出 debug APK（本机零 SDK），别忘了 `network_security_config` 放行你的 IP。
4. 别用 `file://` + fetch，也别上自签名 HTTPS；直接让 WebView 加载 `http://` 地址，同源、无 CORS、无证书问题。
5. 最值得补的是"地址稳定"：固定电脑 IP 或让手机连电脑的 Windows 移动热点，再配二维码，现场就不会翻车。
