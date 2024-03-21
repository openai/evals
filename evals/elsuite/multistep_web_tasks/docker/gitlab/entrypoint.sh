#!/bin/bash

# Modify the GitLab configuration
echo "external_url 'http://gitlab:8023'" >> /etc/gitlab/gitlab.rb

# Reconfigure GitLab
/opt/gitlab/bin/gitlab-ctl reconfigure

# Start GitLab
/opt/gitlab/embedded/bin/runsvdir-start
