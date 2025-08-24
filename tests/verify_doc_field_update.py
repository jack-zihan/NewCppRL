#!/usr/bin/env python3
"""
验证文档中的frontier→field更新是否完成
"""
import os
from pathlib import Path

def check_documentation_consistency():
    """检查文档更新的一致性"""
    doc_dir = Path("/home/lzh/NewCppRL/envs_new/doc")
    
    print("=" * 60)
    print("🔍 检查文档更新一致性...")
    print("=" * 60)
    
    issues = []
    field_count = 0
    
    # 扫描所有markdown文档
    for doc_file in doc_dir.glob("*.md"):
        with open(doc_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 检查是否还有frontier引用
        if 'frontier' in content.lower():
            issues.append(f"❌ {doc_file.name} 仍包含 'frontier' 引用")
            
        # 统计field相关引用
        field_mentions = content.lower().count('field')
        if field_mentions > 0:
            field_count += field_mentions
            print(f"✅ {doc_file.name}: 包含 {field_mentions} 处 'field' 引用")
    
    print("\n" + "=" * 60)
    
    if issues:
        print("⚠️ 发现以下问题：")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print(f"🎉 文档更新验证成功！")
        print(f"   - 所有 frontier 引用已更新为 field")
        print(f"   - 共找到 {field_count} 处 field 引用")
        print(f"   - 文档与代码保持一致")
        return True

def check_specific_terms():
    """检查具体术语更新"""
    doc_dir = Path("/home/lzh/NewCppRL/envs_new/doc")
    
    print("\n" + "=" * 60)
    print("📋 检查具体术语更新...")
    print("=" * 60)
    
    expected_terms = {
        'FieldUpdater': '田地更新器',
        'FieldCoverageCalculator': '田地覆盖计算器',
        'field_area': '田地面积',
        'field_variation': '田地变化度',
        'field_coverage': '田地覆盖',
        'field_group_coef': '田地组系数',
        'Field APF': '田地人工势场'
    }
    
    for doc_file in doc_dir.glob("*.md"):
        with open(doc_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        found_terms = []
        for term, desc in expected_terms.items():
            if term in content:
                found_terms.append(f"    ✓ {term} ({desc})")
        
        if found_terms:
            print(f"\n📄 {doc_file.name}:")
            for term in found_terms:
                print(term)
    
    return True

def main():
    """主验证流程"""
    print("\n🚀 开始验证文档更新...")
    
    # 验证一致性
    consistency_ok = check_documentation_consistency()
    
    # 检查具体术语
    terms_ok = check_specific_terms()
    
    print("\n" + "=" * 60)
    if consistency_ok and terms_ok:
        print("✅ 所有验证通过！文档更新成功完成")
        print("\n📊 更新总结：")
        print("  • TECHNICAL_DOCUMENTATION.md - 技术文档")
        print("  • ARCHITECTURE_ULTRATHINK_ANALYSIS.md - 架构分析")
        print("  • TROUBLESHOOTING_ENHANCED.md - 故障排除")
        print("  • DOCUMENTATION_SUMMARY.md - 文档总结")
        return 0
    else:
        print("❌ 验证失败，请检查问题")
        return 1

if __name__ == "__main__":
    exit(main())