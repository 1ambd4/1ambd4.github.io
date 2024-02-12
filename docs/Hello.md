# Hello World!

终于从 Hugo 叛逃过来哩。

按照[VuePress 教程](https://vuejs.press/zh/guide/introduction.html)操作即可，主题选用 [reco](https://github.com/vuepress-reco/vuepress-theme-reco)，在 `build` 的时候出现了如下的问题：

```js
(undefined) assets/js/styles.bf509504.js from Terser
Error: error:0308010C:digital envelope routines::unsupported
```

虽然中文网络上有说是 nodejs 版本问题，让降级到 16.0，但我觉得这并非问题的根源所在，Google 到了 [StackOverflow](https://stackoverflow.com/questions/74548318/how-to-resolve-error-error0308010cdigital-envelope-routinesunsupported-no) 的这个问题：

> I was able to fix it by providing the following environment argument:
>
> ```sh
> export NODE_OPTIONS=--openssl-legacy-provider
> ```

看到有说也许和 `/etc/ssl/openssl.cnf` 这个配置文件有关，但没细说，先搁下吧。
