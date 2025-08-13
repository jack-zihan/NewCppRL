#!/bin/bash

echo "======================================"
echo "推送到GitHub的脚本"
echo "======================================"

# 显示当前状态
echo -e "\n当前Git状态："
git status --short

echo -e "\n当前分支："
git branch --show-current

echo -e "\n待推送的提交："
git log origin/main..HEAD --oneline

echo -e "\n======================================"
echo "选择推送方式："
echo "======================================"
echo "1. 使用GitHub Personal Access Token (推荐)"
echo "2. 使用SSH (需要配置SSH密钥)"
echo "3. 使用用户名密码"

read -p "请选择 (1/2/3): " choice

case $choice in
    1)
        echo -e "\n请访问 https://github.com/settings/tokens 创建Personal Access Token"
        echo "需要的权限: repo (Full control of private repositories)"
        read -p "请输入您的GitHub用户名: " username
        read -sp "请输入您的Personal Access Token: " token
        echo
        
        # 使用token推送
        git push https://${username}:${token}@github.com/jack-zihan/NewCppRL.git main
        ;;
        
    2)
        echo -e "\n切换到SSH远程URL..."
        git remote set-url origin git@github.com:jack-zihan/NewCppRL.git
        echo "远程URL已更改为SSH格式"
        
        echo -e "\n开始推送..."
        git push origin main
        ;;
        
    3)
        echo -e "\n使用HTTPS推送（需要输入用户名和密码）..."
        git push origin main
        ;;
        
    *)
        echo "无效选择"
        exit 1
        ;;
esac

if [ $? -eq 0 ]; then
    echo -e "\n✅ 推送成功！"
    echo "最新提交已同步到GitHub"
else
    echo -e "\n❌ 推送失败"
    echo "请检查认证信息是否正确"
fi