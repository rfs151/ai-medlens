#!/usr/bin/env bash
# ai-medlens 医疗影像检测 —— 本机启停 / 查看状态
# 部署方式：纯 Docker，两个容器，都是 restart: unless-stopped，随 Docker 开机自启
#   medlens-py   Python 服务（页面 + API），docker compose 管理，本机 :8010（.env 的 MEDLENS_PORT）
#   medlens-cpp  C++ 后端，按 README 用 docker run 单独创建（不归 compose 管），本机 :8081
#
#   ./ctl.sh start | stop | restart | status | logs [py|cpp]
#
# 公网（国内 VPS frp 转发）：http://47.117.29.74:8010/  C++ 后端 :8081
set -euo pipefail
cd "$(dirname "$0")"
CPP=medlens-cpp

health() {
  for u in http://127.0.0.1:8010/api/health http://127.0.0.1:8081/api/health; do
    echo "健康检查 $u -> $(curl -s --noproxy '*' -m 5 -o /dev/null -w '%{http_code}' "$u" || true)"
  done
  echo "公网地址 http://47.117.29.74:8010/"
}

case ${1:-status} in
  start)   docker compose up -d medlens-py; docker start $CPP ;;
  stop)    docker compose stop medlens-py; docker stop $CPP ;;
  restart) docker compose restart medlens-py; docker restart $CPP ;;
  status)
    docker compose ps -a --format 'table {{.Service}}\t{{.Status}}\t{{.Ports}}'
    docker ps -a --filter name=^/$CPP$ --format "$CPP\t{{.Status}}\t{{.Ports}}"
    health ;;
  logs)
    case ${2:-py} in
      cpp) docker logs -f --tail 100 $CPP ;;
      *)   docker compose logs -f --tail 100 medlens-py ;;
    esac ;;
  *) sed -n '2,9p' "$0"; exit 1 ;;
esac
