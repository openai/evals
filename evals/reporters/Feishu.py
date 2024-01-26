import os
from pathlib import Path
import json
import datetime

from typing import Dict, Union, List

import requests

# 时间、实验名、项目、成功体系占比、Protocol、imgkey、Tracking链接、工作流链接
FEISHU_MESSAGE_STRING = \
    '''
{
  "config": {
    "wide_screen_mode": true
  },
  "elements": [
    {
      "fields": [
        {
          "is_short": true,
          "text": {
            "content": "**🕐 时间：**\n%s",
            "tag": "lark_md"
          }
        },
        {
          "is_short": true,
          "text": {
            "content": "**🔢 实验名：**\n%s",
            "tag": "lark_md"
          }
        },
        {
          "is_short": false,
          "text": {
            "content": "",
            "tag": "lark_md"
          }
        },
        {
          "is_short": true,
          "text": {
            "content": "**📋 项目：**\n%s",
            "tag": "lark_md"
          }
        },
        {
          "is_short": true,
          "text": {
            "content": "**📋 成功体系：**\n%s",
            "tag": "lark_md"
          }
        }
      ],
      "tag": "div"
    },
    {
      "fields": [
        {
          "is_short": false,
          "text": {
            "content": "**🕐 Protocol：**\n%s",
            "tag": "lark_md"
          }
        }
      ],
      "tag": "div"
    },
    {
      "alt": {
        "content": "",
        "tag": "plain_text"
      },
      "img_key": "%s",
      "tag": "img",
      "title": {
        "content": "Metrics 汇总：",
        "tag": "lark_md"
      }
    },
    {
      "actions": [
        {
          "tag": "button",
          "text": {
            "content": "跟进处理",
            "tag": "plain_text"
          },
          "type": "primary",
          "value": {
            "key1": "value1"
          }
        },
        {
          "options": [
            {
              "text": {
                "content": "屏蔽10分钟",
                "tag": "plain_text"
              },
              "value": "1"
            },
            {
              "text": {
                "content": "屏蔽30分钟",
                "tag": "plain_text"
              },
              "value": "2"
            },
            {
              "text": {
                "content": "屏蔽1小时",
                "tag": "plain_text"
              },
              "value": "3"
            },
            {
              "text": {
                "content": "屏蔽24小时",
                "tag": "plain_text"
              },
              "value": "4"
            }
          ],
          "placeholder": {
            "content": "暂时屏蔽实验跟踪",
            "tag": "plain_text"
          },
          "tag": "select_static",
          "value": {
            "key": "value"
          }
        }
      ],
      "tag": "action"
    },
    {
      "tag": "hr"
    },
    {
      "tag": "div",
      "text": {
        "content": "📝 [Tracking链接](%s) | 🙋 [工作流链接](%s)",
        "tag": "lark_md"
      }
    }
  ],
  "header": {
    "template": "green",
    "title": {
      "content": "IFD 实验跟踪",
      "tag": "plain_text"
    }
  }
}
'''

FEISHU_MESSAGE = {
    "config": {
        "wide_screen_mode": True
    },
    "elements": [
        {
            "fields": [
                {
                    "is_short": True,
                    "text": {
                        "content": "**🕐 时间：**\n%s",
                        "tag": "lark_md"
                    }
                },
                {
                    "is_short": True,
                    "text": {
                        "content": "**🔢 实验名：**\n%s",
                        "tag": "lark_md"
                    }
                },
                {
                    "is_short": False,
                    "text": {
                        "content": "",
                        "tag": "lark_md"
                    }
                },
                {
                    "is_short": True,
                    "text": {
                        "content": "**📋 项目：**\n%s",
                        "tag": "lark_md"
                    }
                },
                {
                    "is_short": True,
                    "text": {
                        "content": "**📋 成功体系：**\n%s",
                        "tag": "lark_md"
                    }
                }
            ],
            "tag": "div"
        },
        {
            "fields": [
                {
                    "is_short": False,
                    "text": {
                        "content": "**🕐 Protocol：**\n%s",
                        "tag": "lark_md"
                    }
                }
            ],
            "tag": "div"
        },
        {
            "alt": {
                "content": "",
                "tag": "plain_text"
            },
            "img_key": "%s",
            "tag": "img",
            "title": {
                "content": "Metrics 汇总：",
                "tag": "lark_md"
            }
        },
        {
            "actions": [
                {
                    "tag": "button",
                    "text": {
                        "content": "跟进处理",
                        "tag": "plain_text"
                    },
                    "type": "primary",
                    "value": {
                        "key1": "value1"
                    }
                },
                {
                    "options": [
                        {
                            "text": {
                                "content": "屏蔽10分钟",
                                "tag": "plain_text"
                            },
                            "value": "1"
                        },
                        {
                            "text": {
                                "content": "屏蔽30分钟",
                                "tag": "plain_text"
                            },
                            "value": "2"
                        },
                        {
                            "text": {
                                "content": "屏蔽1小时",
                                "tag": "plain_text"
                            },
                            "value": "3"
                        },
                        {
                            "text": {
                                "content": "屏蔽24小时",
                                "tag": "plain_text"
                            },
                            "value": "4"
                        }
                    ],
                    "placeholder": {
                        "content": "暂时屏蔽实验跟踪",
                        "tag": "plain_text"
                    },
                    "tag": "select_static",
                    "value": {
                        "key": "value"
                    }
                }
            ],
            "tag": "action"
        },
        {
            "tag": "hr"
        },
        {
            "tag": "div",
            "text": {
                "content": "📝 [Tracking链接](%s) | 🙋 [工作流链接](%s)",
                "tag": "lark_md"
            }
        }
    ],
    "header": {
        "template": "green",
        "title": {
            "content": "IFD 实验跟踪",
            "tag": "plain_text"
        }
    }
}


class FeishuReporter:
    @staticmethod
    def _get_tenant_token(app_id: str = "cli_a301e6759d32500c", app_secret: str = "uLiHOmf0QOQRkhwymy8AmfHWykMQaMFk"):
        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"

        payload = json.dumps({
            "app_id": app_id,
            "app_secret": app_secret
        })
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        response.raise_for_status()
        data = response.json()
        assert data['code'] == 0
        return data['tenant_access_token']

    @staticmethod
    def _upload_image(file_path, type='image/png', app_id: str = "cli_a301e6759d32500c",
                      app_secret: str = "uLiHOmf0QOQRkhwymy8AmfHWykMQaMFk"):
        url = "https://open.feishu.cn/open-apis/im/v1/images"
        payload = {'image_type': 'message'}
        files = [
            ('image', (Path(file_path).stem, open(file_path, 'rb'), type))
        ]
        token = FeishuReporter._get_tenant_token(app_id=app_id, app_secret=app_secret)
        headers = {
            'Authorization': f'Bearer {token}'
        }
        response = requests.request("POST", url, headers=headers, data=payload, files=files)
        response.raise_for_status()
        data = response.json()
        assert data['code'] == 0
        return data['data']['image_key']

    @staticmethod
    def report_run(feishu_groups: List, experiment_group: str, project: str, success_ratio: str,
                       config_protocol: Dict,
                       imgfile: Union[str, Path], track_url: str, workflow_url: str,
                       app_id: str = "", app_secret: str = ""):
        app_id = os.environ.get("FEISHU_APP_ID", app_id)
        app_secret = os.environ.get("FEISHU_APP_SECRET", app_secret)
        now = datetime.datetime.now()
        img_key = FeishuReporter._upload_image(imgfile, app_id=app_id, app_secret=app_secret)

        message = FEISHU_MESSAGE.copy()

        message["elements"][0]["fields"][0]["text"]["content"] = \
            message["elements"][0]["fields"][0]["text"]["content"] % now.strftime("%Y-%m-%d %H:%M:%S")
        message["elements"][0]["fields"][1]["text"]["content"] = \
            message["elements"][0]["fields"][1]["text"]["content"] % experiment_group
        message["elements"][0]["fields"][3]["text"]["content"] = \
            message["elements"][0]["fields"][3]["text"]["content"] % project
        message["elements"][0]["fields"][4]["text"]["content"] = \
            message["elements"][0]["fields"][4]["text"]["content"] % success_ratio
        message["elements"][1]["fields"][0]["text"]["content"] = \
            message["elements"][1]["fields"][0]["text"]["content"] % json.dumps(config_protocol, indent=4)
        message["elements"][2]["img_key"] = img_key
        message["elements"][5]["text"]["content"] = message["elements"][5]["text"]["content"] % (
        track_url, workflow_url)

        for feishu_group in feishu_groups:
            requests.post(feishu_group,
                          json={"msg_type": "interactive", "card": message})
