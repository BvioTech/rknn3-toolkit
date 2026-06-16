#!/bin/bash
#
# 把 RKNN3 Toolkit Lite 的 aarch64 wheel (cp311/cp312) 打成 deb:
#   /opt/violoop/wheels/rknn3_toolkit_lite-*-cp31{1,2}-*-linux_aarch64.whl
#
# 消费端 (app) 在自己的 venv 里离线安装 (pip 自动挑匹配当前 Python 的 wheel):
#   pip install --no-index --find-links /opt/violoop/wheels rknn3-toolkit-lite
#
# 纯数据包, 不编译, 任意架构主机均可构建 (CI 用 ubuntu-latest)。
#
set -euo pipefail

PACKAGE_NAME="violoop-rknn3-toolkit-lite"
ARCH="arm64"
PY_TAGS=("cp311" "cp312")

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 读版本号
[ -f version ] || { echo "错误: version 文件不存在"; exit 1; }
VERSION=$(tr -d '\n' < version | xargs)
[ -n "$VERSION" ] || { echo "错误: version 文件为空"; exit 1; }

WHEEL_SRC="rknn3-toolkit-lite/packages"
STAGING="build/package"
WHEEL_DEST="${STAGING}/opt/violoop/wheels"

echo "=== 清理旧 staging ==="
rm -rf build
mkdir -p "$WHEEL_DEST" "${STAGING}/DEBIAN"

echo "=== 暂存 wheel (${PY_TAGS[*]}) ==="
for tag in "${PY_TAGS[@]}"; do
    whl=$(ls "$WHEEL_SRC"/rknn3_toolkit_lite-*-"${tag}"-"${tag}"-linux_aarch64.whl 2>/dev/null | head -n1)
    [ -n "$whl" ] || { echo "错误: 找不到 ${tag} 的 aarch64 wheel (在 ${WHEEL_SRC}/)"; exit 1; }
    install -m 0644 "$whl" "$WHEEL_DEST/"
    echo "  + $(basename "$whl")"
done

echo "=== 写 DEBIAN/control (版本 ${VERSION}) ==="
sed "s/^Version:.*/Version: ${VERSION}/" package/DEBIAN/control > "${STAGING}/DEBIAN/control"

echo "=== 构建 deb ==="
dpkg-deb --root-owner-group --build "$STAGING" >/dev/null
PACKAGE_FILE="${PACKAGE_NAME}_${VERSION}_${ARCH}.deb"
mv "${STAGING}.deb" "$PACKAGE_FILE"

echo ""
echo "打包完成: $PACKAGE_FILE ($(du -h "$PACKAGE_FILE" | cut -f1))"
